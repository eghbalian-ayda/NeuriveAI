"""
depth_estimator.py — Stage 0b

Depth Anything V2 Small running at 320x240 for per-frame depth maps.
Provides one relative depth value per head centroid for proximity gating.
Eliminates false positive impacts from 2D projection overlap of players
at different depths.

Target: ≤16ms per frame on RTX 3080+ (≥60fps budget).
"""

from __future__ import annotations
import sys
from pathlib import Path
import cv2
import numpy as np
import torch

# ── Constants ──────────────────────────────────────────────────────────────
# Input resolution — lower = faster. 320x240 is sufficient for centroid sampling.
DEPTH_INPUT_SIZE = 320

# Depth difference threshold for proximity gate.
# Depth map values are normalized 0–1 (relative, not metric).
# 0.15 means heads must be within 15% of the full depth range to be
# considered at the same depth. Tune upward if false negatives appear.
DEPTH_SIMILARITY_THRESH = 0.15

# Model config for Small (vits) variant
_MODEL_CFG = {
    "encoder":      "vits",
    "features":     64,
    "out_channels": [48, 96, 192, 384],
}


class DepthEstimator:
    """
    Depth Anything V2 Small wrapper.
    Call estimate_frame() each frame to get the depth map.
    Call head_depth() to sample depth at a centroid pixel location.
    Call same_depth() to gate proximity detection.
    """

    def __init__(
        self,
        ckpt:   str   = "models/depth/depth_anything_v2_vits.pth",
        device: str   = None,
        input_size: int = DEPTH_INPUT_SIZE,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device     = device
        self.input_size = input_size

        # lazy import — DepthAnythingV2 repo may not be present when --no-depth
        DA_ROOT = Path(__file__).parent.parent / "DepthAnythingV2"
        sys.path.insert(0, str(DA_ROOT))
        from depth_anything_v2.dpt import DepthAnythingV2 as _DA2

        ckpt_path = Path(__file__).parent / ckpt
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Depth Anything V2 weights not found at {ckpt_path}\n"
                f"Download from: https://huggingface.co/depth-anything/"
                f"Depth-Anything-V2-Small\n"
                f"Place at: {ckpt_path}"
            )

        self._model = _DA2(**_MODEL_CFG)
        state = torch.load(str(ckpt_path), map_location=device)
        self._model.load_state_dict(state)
        self._model.to(device).eval()

        # attempt TensorRT FP16 export for maximum speed
        self._try_tensorrt()

        # cache last depth map for head_depth() lookups
        self._last_depth: np.ndarray | None = None
        self._last_h: int = 1
        self._last_w: int = 1

    def _try_tensorrt(self) -> None:
        """
        Attempt to export and cache a TensorRT FP16 engine for ~2× speedup.
        Silently skips if TensorRT is not available.
        """
        try:
            import torch_tensorrt   # noqa: F401
            self._model = torch.compile(
                self._model, backend="tensorrt", options={"enabled_precisions": {torch.float16}}
            )
            print("[DepthEstimator] TensorRT FP16 compilation successful")
        except Exception:
            print("[DepthEstimator] TensorRT not available — using standard PyTorch")

    def estimate_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Run depth estimation on one frame.

        Parameters
        ----------
        frame_bgr : np.ndarray  (H, W, 3)  BGR frame from VideoCapture

        Returns
        -------
        depth_norm : np.ndarray  (H, W)  float32, values in [0, 1]
                     0 = farthest, 1 = closest
                     Resized back to original frame resolution for centroid lookup.
        """
        orig_h, orig_w = frame_bgr.shape[:2]

        # resize to inference resolution
        frame_resized = cv2.resize(
            frame_bgr, (self.input_size, self.input_size),
            interpolation=cv2.INTER_LINEAR
        )

        # infer_image handles normalisation internally
        with torch.no_grad():
            depth_raw = self._model.infer_image(
                frame_resized, input_size=self.input_size
            )   # (input_size, input_size) float32, disparity

        # normalize to [0, 1] — higher = closer
        d_min = depth_raw.min()
        d_max = depth_raw.max()
        if d_max > d_min:
            depth_norm = (depth_raw - d_min) / (d_max - d_min)
        else:
            depth_norm = np.zeros_like(depth_raw)

        # resize back to original frame resolution for pixel-accurate sampling
        depth_full = cv2.resize(
            depth_norm, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
        )

        # cache for head_depth() lookups
        self._last_depth = depth_full
        self._last_h     = orig_h
        self._last_w     = orig_w

        return depth_full

    def head_depth(self, centroid: np.ndarray) -> float:
        """
        Sample depth at a head centroid pixel location.

        Parameters
        ----------
        centroid : (2,) [cx, cy] in pixels (from HeadState)

        Returns
        -------
        float in [0, 1] — relative depth. Returns 0.5 if no depth map cached.
        """
        if self._last_depth is None:
            return 0.5   # neutral — don't gate if depth not yet computed

        cx = int(np.clip(centroid[0], 0, self._last_w - 1))
        cy = int(np.clip(centroid[1], 0, self._last_h - 1))
        return float(self._last_depth[cy, cx])

    def same_depth(self, depth_a: float, depth_b: float) -> bool:
        """
        Return True if two depth values are close enough that the
        corresponding heads could plausibly be in physical contact.

        Uses DEPTH_SIMILARITY_THRESH — heads more than 15% of the
        normalized depth range apart are at different depths and
        cannot collide (eliminates 2D projection false positives).
        """
        return abs(depth_a - depth_b) <= DEPTH_SIMILARITY_THRESH
