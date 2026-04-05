# NeurivAI — Complete Pipeline Implementation Plan
# Unified document: V2 head keypoint pipeline + depth gating + TBI estimation
# For Claude Code

---

## What This Plan Builds

A complete rewrite of the NeurivAI pipeline with three passes:

- **Pass 1 (fast, every frame):** YOLOv8x-pose + Depth Anything V2 Small +
  3 lightweight detectors → find all impact timestamps with depth-aware
  proximity gating to eliminate false positives from 2D projection overlap
- **Pass 2 (targeted):** HybrIK retrospectively on ±0.5s windows around
  confirmed impacts → head rotation matrices → angular velocity profiles →
  BrIC_R / KLC / DAMAGE brain injury metrics
- **Pass 3 (per impact):** Wu et al. CNN → regional brain MPS → logistic
  risk curves → overall TBI probability % + brain heatmap PNG

---

## Dependencies — Install All Before Starting

```bash
# Core
pip install ultralytics          # YOLOv8-pose + ByteTrack
pip install torch torchvision    # all ML backends
pip install scipy numpy opencv-python easydict

# Depth estimation
pip install depth-anything-v2    # or clone below
git clone https://github.com/DepthAnything/Depth-Anything-V2.git ../DepthAnythingV2
cd ../DepthAnythingV2 && pip install -r requirements.txt && cd ../competition
# Download Small weights → place at: models/depth/depth_anything_v2_vits.pth
# From: https://huggingface.co/depth-anything/Depth-Anything-V2-Small

# HybrIK
git clone https://github.com/jeffffffli/HybrIK.git ../HybrIK
cd ../HybrIK && pip install -e . && cd ../competition
# Download weights → models/hybrik/hybrik_hrnet.pth  (from HybrIK README)
# Download SMPL → HybrIK/common/utils/smplpytorch/smplpytorch/native/models/
#                 basicModel_neutral_lbs_10_207_0_v1.0.0.pkl

# TBI visualisation
pip install nilearn matplotlib

# Wu et al. CNN weights (optional — fallback activates automatically if missing)
# https://github.com/SJiLab/Brain-Injury-Model
# Place at: models/wu_cnn/wu_strain_cnn.pt
```

---

## Architecture Overview

```
Video (fixed camera)
        │
        ▼
╔══════════════════════════════════════════════════════════════════╗
║  PASS 1 — runs on every frame                                    ║
║                                                                  ║
║  ┌─────────────────────────┐   ┌──────────────────────────────┐ ║
║  │ Stage 0a                │   │ Stage 0b                     │ ║
║  │ HeadKeypointTracker     │   │ DepthEstimator               │ ║
║  │ YOLOv8x-pose + ByteTrack│   │ Depth Anything V2 Small      │ ║
║  │ → HeadState per person  │   │ 320×240px, TensorRT FP16     │ ║
║  │   {centroid, radius,    │   │ → depth_map (H×W float32)    │ ║
║  │    keypoints, body_box} │   │ → depth per head centroid    │ ║
║  └────────────┬────────────┘   └──────────────┬───────────────┘ ║
║               │                               │                  ║
║               └──────────────┬────────────────┘                  ║
║                              │  HeadState enriched with depth    ║
║              ┌───────────────┼───────────────┐                   ║
║              ▼               ▼               ▼                   ║
║       ┌────────────┐  ┌────────────┐  ┌──────────────────┐      ║
║       │ Stage 1    │  │ Stage 2    │  │ Stage 3          │      ║
║       │ Proximity  │  │ Velocity   │  │ Skull Rotation   │      ║
║       │ Detector   │  │ Anomaly    │  │ Detector         │      ║
║       │ (depth-    │  │ Detector   │  │ (ear-vector      │      ║
║       │  gated)    │  │ (z-score)  │  │  angle change)   │      ║
║       └─────┬──────┘  └─────┬──────┘  └────────┬─────────┘      ║
║             └───────────────┴──────────────────┘                 ║
║                             │                                     ║
║                      ImpactBuffer                                 ║
║                 (merges signals, cooldown)                        ║
║                             │                                     ║
║              if ≥2 stages fire on same pair                       ║
║                      → ImpactEvent recorded                       ║
╚══════════════════════════════════════════════════════════════════╝
                             │
                             ▼
╔══════════════════════════════════════════════════════════════════╗
║  PASS 2 — runs only on confirmed impact windows                  ║
║                                                                  ║
║  HybrIKRetrospective                                             ║
║  reads frame_buffer[impact ± 15 frames]                          ║
║  for each flagged track ID:                                       ║
║    → crop person from frame                                       ║
║    → HybrIK inference → R_head(t) per frame (3×3 matrix)        ║
║    → BrainInjuryProfiler:                                        ║
║        dR = R(t+1) @ R(t).T  →  ω(t) = log(dR)/dt              ║
║        → BrIC_R, KLC, DAMAGE                                     ║
╚══════════════════════════════════════════════════════════════════╝
                             │
                             ▼
╔══════════════════════════════════════════════════════════════════╗
║  PASS 3 — per confirmed impact, per track                        ║
║                                                                  ║
║  StrainEstimator                                                  ║
║    ω(t) rendered as 2D image → Wu et al. CNN                     ║
║    → regional MPS {corpus_callosum, brainstem, thalamus,         ║
║                    white_matter, grey_matter, cerebellum}         ║
║    fallback: DAMAGE × 0.56 + anatomical weights                  ║
║                                                                  ║
║  TBIVisualizer                                                    ║
║    logistic risk curves per region → regional probabilities       ║
║    weighted combination → overall TBI probability %              ║
║    nilearn glass brain heatmap + bar chart → PNG                 ║
╚══════════════════════════════════════════════════════════════════╝
                             │
                             ▼
                    impact_report.json
                    impact_XXXXX_track_N_tbi.png
```

---

## File 1 — `depth_estimator.py`  ← NEW

**Purpose:** Wraps Depth Anything V2 Small. Runs once per frame on the full
frame at 320×240 resolution. Exports one depth value per head centroid.
Designed to not block the Pass 1 loop — target ≤16ms per frame (≥60fps).

**Why 320×240:** You only need one depth sample per head centroid, not
fine-grained structure. Coarser resolution → faster inference → 60fps feasible.
The depth values are relative (not metric) but that is sufficient — you only
need to know if two heads are at meaningfully different depths.

**Depth gate logic:** Two heads are candidates for collision only if their
depth values are within `DEPTH_SIMILARITY_THRESH` of each other (normalized
0–1 scale). Heads at different depths that overlap in 2D are filtered out.

```python
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

# ── Depth Anything V2 import ───────────────────────────────────────────────
DA_ROOT = Path(__file__).parent.parent / "DepthAnythingV2"
sys.path.insert(0, str(DA_ROOT))
from depth_anything_v2.dpt import DepthAnythingV2 as _DA2

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
```

---

## File 2 — `head_tracker.py`  ← MODIFIED from V2 plan

**Changes from V2:** `HeadState` gains a `depth` field. The `track()` generator
now accepts a `depth_estimator` argument and populates `HeadState.depth`
by sampling the depth map at each centroid after it is computed.

```python
"""
head_tracker.py — Stage 0a

Multi-person head keypoint tracker using YOLOv8x-pose + ByteTrack.
Now depth-aware: HeadState includes a relative depth value sampled
from the Depth Anything V2 depth map at each head centroid.
"""

from __future__ import annotations
from dataclasses import dataclass
from collections import deque
import cv2
import numpy as np
from ultralytics import YOLO

_NOSE, _LEFT_EYE, _RIGHT_EYE, _LEFT_EAR, _RIGHT_EAR = 0, 1, 2, 3, 4
HEAD_KP_INDICES = [_NOSE, _LEFT_EYE, _RIGHT_EYE, _LEFT_EAR, _RIGHT_EAR]
KP_CONF_THRESH  = 0.30


@dataclass
class HeadState:
    """All head information for a single person in a single frame."""
    track_id:   int
    frame_idx:  int
    centroid:   np.ndarray   # (2,) [cx, cy] pixels
    radius_px:  float
    keypoints:  np.ndarray   # (5, 3) [x, y, conf]
    body_box:   np.ndarray   # (4,) [x1,y1,x2,y2]
    depth:      float = 0.5  # ← NEW: relative depth [0=far, 1=close]


class HeadKeypointTracker:

    def __init__(
        self,
        source:        str,
        model_name:    str   = "yolov8x-pose.pt",
        kp_conf:       float = KP_CONF_THRESH,
        det_conf:      float = 0.35,
        buffer_frames: int   = 90,
        show:          bool  = False,
    ):
        self.source      = source
        self.kp_conf     = kp_conf
        self.det_conf    = det_conf
        self.show        = show
        self._model      = YOLO(model_name)
        self.frame_buffer: deque[tuple[int, np.ndarray]] = deque(
            maxlen=buffer_frames
        )
        self.kp_history:  dict[int, list[HeadState]] = {}
        self._frame_idx   = 0
        self.fps: float   = 30.0

    @staticmethod
    def _extract_head_state(
        track_id: int, frame_idx: int,
        kps_raw: np.ndarray, body_box: np.ndarray,
        kp_conf_thresh: float,
    ) -> "HeadState | None":
        head_kps     = kps_raw[HEAD_KP_INDICES]
        visible_mask = head_kps[:, 2] > kp_conf_thresh
        if not visible_mask.any():
            return None
        visible_pts = head_kps[visible_mask, :2]
        centroid    = visible_pts.mean(axis=0)
        left_ear_vis  = head_kps[3, 2] > kp_conf_thresh
        right_ear_vis = head_kps[4, 2] > kp_conf_thresh
        if left_ear_vis and right_ear_vis:
            radius_px = max(abs(head_kps[4, 0] - head_kps[3, 0]) / 2.0, 8.0)
        else:
            radius_px = max((body_box[2] - body_box[0]) / 8.0, 8.0)
        return HeadState(
            track_id=track_id, frame_idx=frame_idx,
            centroid=centroid, radius_px=radius_px,
            keypoints=head_kps, body_box=body_box,
            depth=0.5,   # filled in by track() after depth estimation
        )

    def track(self, depth_estimator=None):
        """
        Generator — yields (frame_idx, frame_bgr, list[HeadState]) per frame.

        Parameters
        ----------
        depth_estimator : DepthEstimator | None
            If provided, depth_estimator.estimate_frame() is called each frame
            and HeadState.depth is populated by sampling the result.
            If None, all HeadState.depth values default to 0.5 (no gating).
        """
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open: {self.source}")
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.frame_buffer.append((self._frame_idx, frame.copy()))

            # ── Stage 0b: depth estimation ───────────────────────────────
            # Run before YOLO so depth values are ready for HeadState
            if depth_estimator is not None:
                depth_estimator.estimate_frame(frame)

            # ── Stage 0a: pose detection + tracking ───────────────────────
            results = self._model.track(
                frame, persist=True, tracker="bytetrack.yaml",
                conf=self.det_conf, verbose=False, classes=[0],
            )[0]

            head_states: list[HeadState] = []

            if results.keypoints is not None and results.boxes.id is not None:
                kps_all   = results.keypoints.data.cpu().numpy()
                ids_all   = results.boxes.id.cpu().numpy().astype(int)
                boxes_all = results.boxes.xyxy.cpu().numpy()

                for tid, kps, box in zip(ids_all, kps_all, boxes_all):
                    hs = self._extract_head_state(
                        int(tid), self._frame_idx, kps, box, self.kp_conf
                    )
                    if hs is None:
                        continue

                    # populate depth from cached depth map
                    if depth_estimator is not None:
                        hs.depth = depth_estimator.head_depth(hs.centroid)

                    head_states.append(hs)
                    if tid not in self.kp_history:
                        self.kp_history[tid] = []
                    self.kp_history[tid].append(hs)

            if self.show:
                _draw_heads(frame, head_states)
                cv2.imshow("NeurivAI — Head Tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            yield self._frame_idx, frame, head_states
            self._frame_idx += 1

        cap.release()
        if self.show:
            cv2.destroyAllWindows()

    def get_frame_window(
        self, center_frame: int, half_window: int
    ) -> list[tuple[int, np.ndarray]]:
        result = [(f, fr) for f, fr in self.frame_buffer
                  if abs(f - center_frame) <= half_window]
        return sorted(result, key=lambda x: x[0])


def _draw_heads(frame, states):
    for hs in states:
        cx, cy = int(hs.centroid[0]), int(hs.centroid[1])
        cv2.circle(frame, (cx, cy), int(hs.radius_px), (0, 255, 0), 2)
        cv2.putText(frame, f"#{hs.track_id} d={hs.depth:.2f}",
                    (cx - 10, cy - int(hs.radius_px) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
```

---

## File 3 — `proximity_detector.py`  ← MODIFIED from V2 plan

**Changes from V2:** Adds depth gate. Uses `DepthEstimator.same_depth()` to
reject pairs where the two heads are at meaningfully different depths.
Adds `size_ratio` gate as a secondary fast check before depth lookup.

```python
"""
proximity_detector.py — Stage 1

Depth-gated normalized head proximity detector.
Two guards against false positives from 2D projection:
  1. Size ratio gate  — very different apparent head sizes → different depths
  2. Depth gate       — Depth Anything V2 depth values differ by >15%
Both must pass for a proximity hit to be emitted.
"""

from __future__ import annotations
import numpy as np
from itertools import combinations
from head_tracker import HeadState

PROXIMITY_THRESHOLD  = 2.5    # head-radius units
SIZE_RATIO_MAX       = 1.8    # max ratio of larger/smaller head radius
DEPTH_SIMILARITY_THRESH = 0.15  # must match depth_estimator.py constant


class ProximityDetector:

    def __init__(
        self,
        threshold:        float = PROXIMITY_THRESHOLD,
        size_ratio_max:   float = SIZE_RATIO_MAX,
        depth_thresh:     float = DEPTH_SIMILARITY_THRESH,
    ):
        self.threshold      = threshold
        self.size_ratio_max = size_ratio_max
        self.depth_thresh   = depth_thresh

    def detect(self, states: list[HeadState]) -> list[dict]:
        """
        Returns list of proximity hits. Each hit:
            id_a, id_b    : track IDs
            dist_norm     : distance in head-radius units
            score         : 0–1 proximity score
            depth_a/b     : depth values of each head
        """
        results = []
        for a, b in combinations(states, 2):

            # ── Guard 1: size ratio (fast, no depth lookup needed) ────────
            ratio = max(a.radius_px, b.radius_px) / max(
                min(a.radius_px, b.radius_px), 1.0
            )
            if ratio > self.size_ratio_max:
                continue   # heads too different in size → different depths

            # ── Guard 2: depth gate ───────────────────────────────────────
            if abs(a.depth - b.depth) > self.depth_thresh:
                continue   # heads at different depths → 2D projection artifact

            # ── Proximity score ───────────────────────────────────────────
            dist_px   = float(np.linalg.norm(a.centroid - b.centroid))
            mean_r    = (a.radius_px + b.radius_px) / 2.0
            dist_norm = dist_px / max(mean_r, 1.0)

            if dist_norm < self.threshold:
                score = max(0.0, 1.0 - dist_norm / self.threshold)
                results.append({
                    "id_a":      a.track_id,
                    "id_b":      b.track_id,
                    "dist_norm": round(dist_norm, 4),
                    "score":     round(score, 4),
                    "depth_a":   round(a.depth, 3),
                    "depth_b":   round(b.depth, 3),
                })

        return sorted(results, key=lambda x: x["dist_norm"])
```

---

## File 4 — `velocity_detector.py`  ← UNCHANGED from V2 plan

Copy exactly from V2 plan. No changes needed.

---

## File 5 — `skull_rotation_detector.py`  ← UNCHANGED from V2 plan

Copy exactly from V2 plan. No changes needed.

---

## File 6 — `impact_buffer.py`  ← UNCHANGED from V2 plan

Copy exactly from V2 plan. No changes needed.

---

## File 7 — `hybrik_retrospective.py`  ← UNCHANGED from V2 plan

Copy exactly from V2 plan. No changes needed.

---

## File 8 — `brain_injury_profiler.py`  ← UNCHANGED from V2 plan

Copy exactly from V2 plan. No changes needed.

---

## File 9 — `strain_estimator.py`  ← UNCHANGED from TBI plan

Copy exactly from TBI plan. No changes needed.

---

## File 10 — `tbi_visualizer.py`  ← UNCHANGED from TBI plan

Copy exactly from TBI plan. No changes needed.

---

## File 11 — `track_video.py`  ← COMPLETE REWRITE (unified)

This is the only file that changes significantly from what either plan
individually specified. It now orchestrates all three passes and both
new modules (depth + TBI).

```python
"""
track_video.py — Unified pipeline entry point

Three-pass processing:
  Pass 1: YOLOv8x-pose + Depth Anything V2 + 3 lightweight detectors
          → depth-gated impact detection → impact timestamps
  Pass 2: HybrIK retrospective on ±HALF_WINDOW frames
          → rotation matrices → BrIC_R / KLC / DAMAGE
  Pass 3: Wu et al. CNN → regional MPS → TBI probability + heatmap PNG

Usage:
    python track_video.py --video path/to/video.mp4 [--window 15] [--show]
    python track_video.py --video path/to/video.mp4 --no-depth   # skip depth gate
"""

import argparse
import json
import numpy as np
from pathlib import Path

from depth_estimator         import DepthEstimator
from head_tracker            import HeadKeypointTracker
from proximity_detector      import ProximityDetector
from velocity_detector       import KeypointVelocityDetector
from skull_rotation_detector import SkullRotationDetector
from impact_buffer           import ImpactBuffer, ImpactEvent
from hybrik_retrospective    import HybrIKRetrospective
from brain_injury_profiler   import BrainInjuryProfiler
from strain_estimator        import StrainEstimator
from tbi_visualizer          import TBIVisualizer


HALF_WINDOW = 15   # ±frames around impact for HybrIK (0.5s at 30fps)


def run(
    video_path: str,
    half_window: int  = HALF_WINDOW,
    show:        bool = False,
    use_depth:   bool = True,    # set False to skip depth gate for debugging
):
    print(f"\n{'='*60}")
    print(f"NeurivAI — Complete Head Impact Pipeline")
    print(f"Video      : {video_path}")
    print(f"Depth gate : {'ON' if use_depth else 'OFF'}")
    print(f"{'='*60}\n")

    out_dir = str(Path(video_path).parent)

    # ── initialise all modules ─────────────────────────────────────────────

    # depth estimator (optional — skip with --no-depth for debugging)
    depth_est = None
    if use_depth:
        try:
            depth_est = DepthEstimator()
            print("[Init] Depth Anything V2 Small loaded\n")
        except FileNotFoundError as e:
            print(f"[Init] WARNING: {e}")
            print("[Init] Continuing without depth gate — "
                  "proximity may have false positives\n")

    tracker   = HeadKeypointTracker(
        source=video_path,
        show=show,
        buffer_frames=half_window * 2 + 10,
    )
    proximity  = ProximityDetector()
    velocity   = KeypointVelocityDetector()
    skull_rot  = SkullRotationDetector(fps=30.0)
    imp_buf    = ImpactBuffer()

    # pass 2+3 modules — load once, reuse for all events
    hybrik     = HybrIKRetrospective()
    profiler   = BrainInjuryProfiler()
    strain_est = StrainEstimator()
    tbi_viz    = TBIVisualizer(output_dir=out_dir)

    # ── PASS 1: depth-gated lightweight detection ──────────────────────────
    print("[Pass 1] Head tracking + depth-gated impact detection...")

    for frame_idx, frame, head_states in tracker.track(
        depth_estimator=depth_est
    ):
        if frame_idx == 0:
            skull_rot.fps = tracker.fps

        prox_hits  = proximity.detect(head_states)
        vel_hits   = velocity.detect(head_states)
        skull_hits = skull_rot.detect(head_states)

        new_events = imp_buf.process_frame(
            frame_idx, prox_hits, vel_hits, skull_hits
        )

        for ev in new_events:
            print(
                f"  [IMPACT] frame={ev.frame_idx:05d} | "
                f"tracks={ev.track_ids} | conf={ev.confidence:.3f} | "
                f"stages={ev.stages}"
            )

    all_events = imp_buf.events
    print(f"\n[Pass 1 complete] {len(all_events)} impact event(s) found.\n")

    if not all_events:
        print("No impacts detected. Done.")
        return []

    # ── PASSES 2 + 3: HybrIK → brain injury → TBI estimation ─────────────
    print("[Pass 2+3] HybrIK + brain injury + TBI estimation...")

    reports = []

    for ev in all_events:
        print(f"\n  Event @ frame {ev.frame_idx} — tracks {ev.track_ids}")

        frame_window = tracker.get_frame_window(ev.frame_idx, half_window)
        if not frame_window:
            print("    ⚠ No buffered frames — event too early in video")
            continue

        for tid in ev.track_ids:

            # ── Pass 2: HybrIK → rotation matrices → velocity profile ─────
            rot_dict = hybrik.process_event(frame_window, tid, tracker)
            if not rot_dict:
                print(f"    ⚠ track {tid}: no HybrIK output")
                continue

            report = profiler.profile(
                rot_dict, tracker.fps, tid, ev.frame_idx
            )

            if "error" in report:
                print(f"    track {tid}: {report['error']}")
                reports.append(report)
                continue

            print(
                f"    track {tid}: "
                f"ω_peak={report['omega_peak_rad_s']:.1f} rad/s | "
                f"BrIC_R={report['bric_r']:.3f} ({report['bric_r_risk']}) | "
                f"KLC={report['klc_rot_rad_s']:.1f} rad/s | "
                f"DAMAGE={report['damage']:.4f} | "
                f"RISK={report['risk_summary']}"
            )

            # ── Pass 3: ω(t) → MPS → TBI probability + heatmap ───────────
            omega_np = np.array(report["omega_xyz"])

            regional_mps = strain_est.estimate(
                omega        = omega_np,
                damage_score = report.get("damage", 0.0),
            )

            tbi_result = tbi_viz.visualize(
                regional_mps = regional_mps,
                track_id     = tid,
                event_frame  = ev.frame_idx,
            )

            # merge TBI results into report
            report["tbi_probability_pct"]  = tbi_result["overall_tbi_pct"]
            report["tbi_risk_label"]       = tbi_result["overall_risk_label"]
            report["regional_tbi_probs"]   = tbi_result["regional_probs"]
            report["tbi_figure"]           = tbi_result["figure_path"]

            print(
                f"    ↳ TBI: {tbi_result['overall_tbi_pct']:.1f}% "
                f"[{tbi_result['overall_risk_label']}] — "
                f"{tbi_result['figure_path']}"
            )

            reports.append(report)

    # ── save unified JSON report ───────────────────────────────────────────
    out_path = Path(video_path).with_suffix(".impact_report.json")
    with open(out_path, "w") as f:
        json.dump({
            "events": [
                {
                    "frame":      ev.frame_idx,
                    "tracks":     ev.track_ids,
                    "confidence": ev.confidence,
                    "stages":     ev.stages,
                    "details":    ev.details,
                }
                for ev in all_events
            ],
            "profiles": reports,
        }, f, indent=2)

    print(f"\n[Done] Report → {out_path}")
    return reports


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",    required=True)
    ap.add_argument("--window",   type=int, default=HALF_WINDOW)
    ap.add_argument("--show",     action="store_true")
    ap.add_argument("--no-depth", action="store_true",
                    help="Disable depth gate (for debugging / no GPU)")
    args = ap.parse_args()
    run(
        video_path  = args.video,
        half_window = args.window,
        show        = args.show,
        use_depth   = not args.no_depth,
    )
```

---

## Final File Structure

```
competition/
│
│  ── CORE PIPELINE ──
├── track_video.py                ← UNIFIED ENTRY POINT (three passes)
├── depth_estimator.py            ← NEW  (Stage 0b: Depth Anything V2 Small)
├── head_tracker.py               ← MODIFIED  (HeadState gains depth field)
├── proximity_detector.py         ← MODIFIED  (depth gate + size ratio gate)
├── velocity_detector.py          ← from V2 plan, unchanged
├── skull_rotation_detector.py    ← from V2 plan, unchanged
├── impact_buffer.py              ← from V2 plan, unchanged
├── hybrik_retrospective.py       ← from V2 plan, unchanged
├── brain_injury_profiler.py      ← from V2 plan, unchanged
│
│  ── TBI ESTIMATION ──
├── strain_estimator.py           ← from TBI plan, unchanged
├── tbi_visualizer.py             ← from TBI plan, unchanged
│
│  ── DELETED from original pipeline ──
├── helmet_tracker.py             ← DELETE
├── iou_detector.py               ← DELETE
├── hot_detector.py               ← DELETE
├── impact_detector.py            ← DELETE
│
└── models/
    ├── depth/
    │   └── depth_anything_v2_vits.pth   ← download (Apache 2.0)
    │       https://huggingface.co/depth-anything/Depth-Anything-V2-Small
    ├── hybrik/
    │   └── hybrik_hrnet.pth             ← download from HybrIK README
    └── wu_cnn/
        └── wu_strain_cnn.pt             ← download from Ji Lab GitHub
                                            (fallback if missing)

../DepthAnythingV2/                       ← git clone + pip install -r
../HybrIK/                               ← git clone + pip install -e .
```

---

## Per-Frame Compute Budget (Pass 1, RTX 3080)

| Stage | Model | Resolution | Target |
|---|---|---|---|
| Stage 0a | YOLOv8x-pose + ByteTrack | native | ~20ms |
| Stage 0b | Depth Anything V2 Small | 320×240 | ~10ms |
| Stage 1 | ProximityDetector | — | <1ms |
| Stage 2 | VelocityDetector | — | <1ms |
| Stage 3 | SkullRotationDetector | — | <1ms |
| **Total** | | | **~32ms ≈ 31fps** |

With TensorRT FP16 on the depth model: ~22ms → ~45fps.
At 320×240 + TensorRT + RTX 4080: ≥60fps achievable.

Pass 2+3 run only on confirmed events — not in the frame loop.

---

## Key Architectural Decisions for Claude Code

**Why depth_estimator runs before YOLO in the frame loop:**
Both run on the same frame. The depth map must be cached before
`_extract_head_state()` so that `head_depth(centroid)` can look it up
immediately. The order in `track()` is: `estimate_frame()` → YOLO →
populate `HeadState.depth` for each person.

**Why depth gate uses relative depth, not metric:**
Depth Anything V2 Small outputs disparity (relative depth), not metres.
That is sufficient — you only need to know if two heads are at similar
relative depths in the scene, not their exact distance. The 0.15 threshold
(15% of normalized range) is tunable via `DEPTH_SIMILARITY_THRESH`.

**Why size ratio is checked before depth:**
The size ratio check is a free arithmetic operation — no depth lookup needed.
It catches the most obvious cases (player 5m away vs. player 1m away will
have very different head radii) before spending time on depth sampling.

**Why `--no-depth` flag exists:**
During debugging or on machines without a GPU for depth estimation, it is
useful to run the pipeline with depth gating disabled. The flag makes this
a CLI option without requiring code changes.

**Why Pass 2 and Pass 3 are in the same loop:**
Both operate per-impact-event per-track. Combining them avoids a third
video scan. The order within the loop is strict:
HybrIK → profiler → strain_estimator → tbi_visualizer.
Each step feeds the next with no branching.

**Why `use_depth=True` is the default:**
Depth gating is the correct behavior. The `--no-depth` flag is an escape
hatch, not the normal operating mode. Claude Code should implement it
but not document it prominently in any user-facing output.
