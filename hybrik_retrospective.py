"""
hybrik_retrospective.py — Stage 4 (on-demand)

Runs HybrIK ONLY on buffered frames around a confirmed impact event.
This is the compute-expensive step — but it only ever runs on
~30-90 frames total across the entire video regardless of video length.

HybrIK outputs:
  pred_theta_mats: (1, 24*9) — 24 rotation matrices flattened
  SMPL joint 15 = head joint
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np
import torch
import cv2

HYBRIK_ROOT = Path(__file__).parent.parent.parent / "HybrIK"
sys.path.insert(0, str(HYBRIK_ROOT))

# Defer HybrIK imports to runtime — allows Pass 1 to run without HybrIK installed
try:
    from hybrik.models import builder as hybrik_builder
    from hybrik.utils.config import update_config
    _HYBRIK_AVAILABLE = True
except ImportError:
    _HYBRIK_AVAILABLE = False

HEAD_JOINT_IDX = 15   # SMPL 24-joint skeleton: joint 15 = head


class HybrIKRetrospective:
    """
    Loaded once at startup. Call process_event() for each confirmed impact.
    """

    def __init__(
        self,
        ckpt: str = "models/hybrik/hybrik_hrnet.pth",
        cfg_file: str = None,
        device: str = None,
    ):
        if not _HYBRIK_AVAILABLE:
            raise RuntimeError(
                "HybrIK is not installed. Clone and install it:\n"
                "  git clone https://github.com/Jeff-sjtu/HybrIK.git ../HybrIK\n"
                "  cd ../HybrIK && pip install -e ."
            )
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if cfg_file is None:
            cfg_file = str(
                HYBRIK_ROOT / "configs" /
                "256x192_adam_lr1e-3-hrw48_cam_2x_w_pw3d_3dhp.yaml"
            )

        # HybrIK uses relative paths internally — must run from its root
        _orig_cwd = os.getcwd()
        os.chdir(str(HYBRIK_ROOT))
        try:
            cfg = update_config(cfg_file)
            self.model = hybrik_builder.build_sppe(cfg.MODEL)
        finally:
            os.chdir(_orig_cwd)

        # search for the checkpoint: project-relative, then HYBRIK_ROOT variants
        ckpt_path = Path(_orig_cwd) / ckpt
        if not ckpt_path.exists():
            for candidate in [
                HYBRIK_ROOT / "hybrik" / "models" / Path(ckpt).name,
                HYBRIK_ROOT / "hybrik" / "models" / "pretrained_hrnet.pth",
                HYBRIK_ROOT / "pretrained_models" / "hybrik_hrnet.pth",
            ]:
                if candidate.exists():
                    ckpt_path = candidate
                    break
        state      = torch.load(str(ckpt_path), map_location=device)
        state_dict = state.get("model", state.get("state_dict", state))
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(device).eval()
        print("HybrIK (retrospective) loaded.")

        self._mean = np.array([0.485, 0.456, 0.406], np.float32)
        self._std  = np.array([0.229, 0.224, 0.225], np.float32)

    def _crop_and_preprocess(
        self,
        frame_bgr: np.ndarray,
        body_box: np.ndarray,   # [x1,y1,x2,y2]
        pad: float = 0.15,
    ) -> torch.Tensor | None:
        """Crop person from frame, pad, resize to 256×256, normalize."""
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = body_box

        # add padding
        bw = x2 - x1
        bh = y2 - y1
        x1 = max(0, x1 - pad * bw)
        y1 = max(0, y1 - pad * bh)
        x2 = min(w, x2 + pad * bw)
        y2 = min(h, y2 + pad * bh)

        crop = frame_bgr[int(y1):int(y2), int(x1):int(x2)]
        if crop.size == 0:
            return None

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_res = cv2.resize(crop_rgb, (256, 256))
        inp      = (crop_res / 255.0 - self._mean) / self._std
        return torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).float()

    def process_event(
        self,
        frame_window: list[tuple[int, np.ndarray]],  # (frame_idx, BGR)
        track_id: int,
        tracker,    # HeadKeypointTracker — for body_box history
    ) -> dict[int, np.ndarray]:
        """
        Run HybrIK on each frame in frame_window for the given track_id.

        Returns
        -------
        dict[frame_idx → rotation_matrix (3,3)]
            Head rotation matrix in camera space per frame.
        """
        rot_by_frame: dict[int, np.ndarray] = {}

        for fidx, frame_bgr in frame_window:
            # find the HeadState for this track and frame
            history = tracker.kp_history.get(track_id, [])
            hs = next((h for h in history if h.frame_idx == fidx), None)
            if hs is None:
                continue

            inp = self._crop_and_preprocess(frame_bgr, hs.body_box)
            if inp is None:
                continue

            inp = inp.to(self.device)

            with torch.no_grad():
                output = self.model(inp)

            # pred_theta_mats: (1, 24*9) — 24 rotation matrices flattened
            theta    = output.pred_theta_mats.cpu().numpy()   # (1, 216)
            rot_mats = theta.reshape(1, 24, 3, 3)
            head_rot = rot_mats[0, HEAD_JOINT_IDX]            # (3, 3)
            rot_by_frame[fidx] = head_rot

        return rot_by_frame
