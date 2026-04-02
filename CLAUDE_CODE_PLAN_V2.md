# NeurivAI v2 — Head Keypoint Impact Pipeline
# Claude Code Implementation Plan

## Overview

Complete rewrite of the NeurivAI pipeline around head keypoints instead of
bounding boxes. Designed for:
- Fixed camera, 10+ people, single view
- Real-time-capable demo on pre-recorded video
- Three lightweight detection stages running every frame
- HybrIK runs retrospectively ONLY on confirmed impact windows
- Brain injury metrics (BrIC_R, KLC, DAMAGE) computed from HybrIK rotation matrices

**Strategy:** Two-pass processing on the video file.
- Pass 1: YOLOv8x-pose + 3 lightweight stages → find all impact timestamps (fast)
- Pass 2: HybrIK on ±W frames around each impact for flagged IDs only (targeted)

---

## Dependencies to Install First

```bash
pip install ultralytics          # YOLOv8-pose + ByteTrack
pip install torch torchvision    # HybrIK backend
pip install scipy numpy opencv-python easydict

# Clone and install HybrIK
git clone https://github.com/jeffffffli/HybrIK.git ../HybrIK
cd ../HybrIK && pip install -e . && cd ../competition

# Download pretrained HybrIK weights (HRNet-W48 with camera prediction)
# From HybrIK README → Google Drive → place at:
# competition/models/hybrik/hybrik_hrnet.pth
# Also download SMPL neutral model → place at:
# HybrIK/common/utils/smplpytorch/smplpytorch/native/models/
#   basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
```

---

## File 1 — `head_tracker.py`

**Purpose:** Stage 0. Runs YOLOv8x-pose on every frame to extract 5 head keypoints
per person, computes head centroid + radius, and assigns persistent track IDs via
ByteTrack. Maintains a rolling frame buffer and keypoint history buffer.

**COCO head keypoint indices used:**
- 0 = nose
- 1 = left_eye
- 2 = right_eye
- 3 = left_ear
- 4 = right_ear

**Head centroid:** mean of all visible keypoints (conf > 0.3)
**Head radius:** half the mean horizontal span (right_ear_x - left_ear_x),
               falling back to bbox width/4 if ears not visible

```python
"""
head_tracker.py — Stage 0
Multi-person head keypoint tracker using YOLOv8x-pose + ByteTrack.
Maintains rolling frame + keypoint buffers for retrospective HybrIK pass.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
import cv2
import numpy as np
from ultralytics import YOLO


# COCO keypoint indices relevant to head
_NOSE      = 0
_LEFT_EYE  = 1
_RIGHT_EYE = 2
_LEFT_EAR  = 3
_RIGHT_EAR = 4
HEAD_KP_INDICES = [_NOSE, _LEFT_EYE, _RIGHT_EYE, _LEFT_EAR, _RIGHT_EAR]

# visibility threshold for a keypoint to count
KP_CONF_THRESH = 0.30


@dataclass
class HeadState:
    """All head information for a single person in a single frame."""
    track_id:   int
    frame_idx:  int
    centroid:   np.ndarray          # (2,) [cx, cy] in pixels
    radius_px:  float               # head radius proxy in pixels
    keypoints:  np.ndarray          # (5, 3) [x, y, conf] for HEAD_KP_INDICES
    body_box:   np.ndarray          # (4,) [x1,y1,x2,y2] full body bbox for HybrIK crop


class HeadKeypointTracker:
    """
    Runs YOLOv8x-pose each frame, extracts head keypoints, assigns ByteTrack IDs.
    Buffers frames and keypoint histories for retrospective processing.
    """

    def __init__(
        self,
        source: str,
        model_name: str = "yolov8x-pose.pt",   # largest for crowd accuracy
        kp_conf: float  = KP_CONF_THRESH,
        det_conf: float = 0.35,
        buffer_frames: int = 90,                # rolling buffer: 3s at 30fps
        show: bool = False,
    ):
        self.source       = source
        self.kp_conf      = kp_conf
        self.det_conf     = det_conf
        self.show         = show
        self.buffer_size  = buffer_frames

        # YOLOv8x-pose — downloads automatically on first run
        self._model = YOLO(model_name)

        # Rolling buffers
        # frame_buffer[i] = BGR frame at absolute frame index i
        self.frame_buffer: deque[tuple[int, np.ndarray]] = deque(
            maxlen=buffer_frames
        )
        # kp_history[track_id] = list of HeadState, chronological
        self.kp_history:   dict[int, list[HeadState]] = {}

        self._frame_idx = 0
        self.fps: float = 30.0     # updated from video metadata

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_head_state(
        track_id: int,
        frame_idx: int,
        kps_raw: np.ndarray,    # (17, 3) full body keypoints [x, y, conf]
        body_box: np.ndarray,   # (4,) [x1,y1,x2,y2]
        kp_conf_thresh: float,
    ) -> HeadState | None:
        """
        Extract HeadState from YOLO keypoint output for one person.
        Returns None if no head keypoints are visible.
        """
        head_kps = kps_raw[HEAD_KP_INDICES]   # (5, 3)

        # visible = confidence above threshold
        visible_mask = head_kps[:, 2] > kp_conf_thresh
        if not visible_mask.any():
            return None

        visible_pts = head_kps[visible_mask, :2]   # (N, 2)
        centroid    = visible_pts.mean(axis=0)      # (2,)

        # radius from ear span if both ears visible
        left_ear_vis  = head_kps[3, 2] > kp_conf_thresh
        right_ear_vis = head_kps[4, 2] > kp_conf_thresh
        if left_ear_vis and right_ear_vis:
            span      = abs(head_kps[4, 0] - head_kps[3, 0])
            radius_px = max(span / 2.0, 8.0)
        else:
            # fallback: use body box width as proxy
            radius_px = max((body_box[2] - body_box[0]) / 8.0, 8.0)

        return HeadState(
            track_id  = track_id,
            frame_idx = frame_idx,
            centroid  = centroid,
            radius_px = radius_px,
            keypoints = head_kps,          # (5, 3)
            body_box  = body_box,
        )

    # ── main generator ─────────────────────────────────────────────────────────

    def track(self):
        """
        Generator — yields (frame_idx, frame_bgr, list[HeadState]) per frame.
        Also populates self.frame_buffer and self.kp_history.
        """
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open: {self.source}")

        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # buffer raw frame for HybrIK retrospective pass
            self.frame_buffer.append((self._frame_idx, frame.copy()))

            # run YOLOv8-pose with ByteTrack
            results = self._model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=self.det_conf,
                verbose=False,
                classes=[0],    # person only
            )[0]

            head_states: list[HeadState] = []

            if results.keypoints is not None and results.boxes.id is not None:
                kps_all   = results.keypoints.data.cpu().numpy()   # (N, 17, 3)
                ids_all   = results.boxes.id.cpu().numpy().astype(int)
                boxes_all = results.boxes.xyxy.cpu().numpy()        # (N, 4)

                for i, (tid, kps, box) in enumerate(
                    zip(ids_all, kps_all, boxes_all)
                ):
                    hs = self._extract_head_state(
                        int(tid), self._frame_idx, kps, box, self.kp_conf
                    )
                    if hs is None:
                        continue

                    head_states.append(hs)

                    # accumulate keypoint history
                    if tid not in self.kp_history:
                        self.kp_history[tid] = []
                    self.kp_history[tid].append(hs)

            if self.show:
                _draw_heads(frame, head_states)
                cv2.imshow("Head Tracker", frame)
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
        """
        Retrieve buffered frames in [center_frame - half_window,
        center_frame + half_window] from the rolling buffer.
        Returns list of (frame_idx, BGR frame).
        """
        result = []
        for fidx, frame in self.frame_buffer:
            if abs(fidx - center_frame) <= half_window:
                result.append((fidx, frame))
        return sorted(result, key=lambda x: x[0])


def _draw_heads(frame: np.ndarray, states: list[HeadState]) -> None:
    """Debug visualisation — draw head centroids and keypoints."""
    for hs in states:
        cx, cy = int(hs.centroid[0]), int(hs.centroid[1])
        r = int(hs.radius_px)
        cv2.circle(frame, (cx, cy), r, (0, 255, 0), 2)
        cv2.putText(frame, str(hs.track_id), (cx - 10, cy - r - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        for kp in hs.keypoints:
            if kp[2] > KP_CONF_THRESH:
                cv2.circle(frame, (int(kp[0]), int(kp[1])), 3, (255, 0, 0), -1)
```

---

## File 2 — `proximity_detector.py`

**Purpose:** Stage 1. Computes pairwise normalized head-to-head distance each frame.
Scale-invariant: threshold is in units of head radii, not pixels.

```python
"""
proximity_detector.py — Stage 1
Pairwise normalized head proximity. Scale-invariant across frame.
"""

from __future__ import annotations
import numpy as np
from itertools import combinations
from head_tracker import HeadState


# Impact threshold: heads touching ≈ centroids 2 radii apart
# Set to 2.5 to catch imminent contact + glancing blows
PROXIMITY_THRESHOLD = 2.5


class ProximityDetector:
    """
    For each pair of heads, compute:
        dist_norm = euclidean(centroid_A, centroid_B) / mean(radius_A, radius_B)

    Fires if dist_norm < PROXIMITY_THRESHOLD.
    Score = max(0, 1 - dist_norm / PROXIMITY_THRESHOLD)
            → 1.0 when centroids coincide, 0.0 at threshold.
    """

    def __init__(self, threshold: float = PROXIMITY_THRESHOLD):
        self.threshold = threshold

    def detect(
        self, states: list[HeadState]
    ) -> list[dict]:
        """
        Parameters
        ----------
        states : list of HeadState for current frame

        Returns
        -------
        list of dicts:
            id_a       : int
            id_b       : int
            dist_norm  : float   (dimensionless, in head-radius units)
            score      : float   (0–1, higher = closer)
        """
        results = []
        for a, b in combinations(states, 2):
            dist_px   = np.linalg.norm(a.centroid - b.centroid)
            mean_r    = (a.radius_px + b.radius_px) / 2.0
            dist_norm = dist_px / max(mean_r, 1.0)

            if dist_norm < self.threshold:
                score = max(0.0, 1.0 - dist_norm / self.threshold)
                results.append({
                    "id_a":      a.track_id,
                    "id_b":      b.track_id,
                    "dist_norm": round(dist_norm, 4),
                    "score":     round(score, 4),
                })

        return sorted(results, key=lambda x: x["dist_norm"])
```

---

## File 3 — `velocity_detector.py`

**Purpose:** Stage 2. Z-score anomaly on head centroid displacement.
Uses head keypoint centroid (not box centroid) — more anatomically stable.

```python
"""
velocity_detector.py — Stage 2
Z-score velocity anomaly detector on head keypoint centroids.
Fixed camera: no ego-motion compensation needed.
"""

from __future__ import annotations
from collections import deque
import numpy as np
from head_tracker import HeadState


class KeypointVelocityDetector:
    """
    Per-track rolling velocity baseline. Flags sudden displacement spikes.

    velocity(t) = ||centroid(t) - centroid(t-1)||   [pixels/frame]
    z(t) = |v(t) - mean(v_window)| / std(v_window)
    Fires if z > z_thresh AND min_history frames seen.
    """

    def __init__(
        self,
        window:      int   = 10,    # rolling baseline window (frames)
        z_thresh:    float = 3.5,   # z-score threshold
        min_history: int   = 6,     # frames needed before firing
    ):
        self.window      = window
        self.z_thresh    = z_thresh
        self.min_history = min_history

        # dict[track_id → deque of (centroid, velocity)]
        self._centroids: dict[int, deque] = {}
        self._velocities: dict[int, deque] = {}

    def detect(self, states: list[HeadState]) -> list[dict]:
        """
        Returns list of anomalous tracks:
            id       : int
            velocity : float  [px/frame]
            z_score  : float
        """
        results = []
        seen_ids = set()

        for hs in states:
            tid = hs.track_id
            seen_ids.add(tid)

            if tid not in self._centroids:
                self._centroids[tid]  = deque(maxlen=self.window + 1)
                self._velocities[tid] = deque(maxlen=self.window)

            self._centroids[tid].append(hs.centroid.copy())

            if len(self._centroids[tid]) < 2:
                continue

            # latest velocity
            v = float(np.linalg.norm(
                self._centroids[tid][-1] - self._centroids[tid][-2]
            ))
            self._velocities[tid].append(v)

            if len(self._velocities[tid]) < self.min_history:
                continue

            vels = np.array(self._velocities[tid])
            mu   = vels[:-1].mean()
            sig  = vels[:-1].std() + 1e-6   # avoid /0
            z    = abs(v - mu) / sig

            if z > self.z_thresh:
                results.append({
                    "id":       tid,
                    "velocity": round(v, 3),
                    "z_score":  round(float(z), 3),
                })

        return results
```

---

## File 4 — `skull_rotation_detector.py`

**Purpose:** Stage 3. Tracks the ear-to-ear (or eye-to-eye) orientation vector per
track. A sudden change in this angle is a direct 2D signature of head snap — the
most specific head-impact indicator computable from keypoints alone.

No ML required. Runs in microseconds. This replaces HOT entirely.

```python
"""
skull_rotation_detector.py — Stage 3
Tracks 2D head orientation vector and detects sudden angular velocity.
The ear-to-ear vector directly encodes head rotation in the image plane.
A sudden change = head snap = impact signature.

Angular velocity threshold ~ 5 rad/s at 30fps corresponds to:
  5 rad/s × (1/30)s = 0.167 rad ≈ 9.5° per frame
This is well above normal voluntary head movement.
"""

from __future__ import annotations
from collections import deque
import numpy as np
from head_tracker import HeadState


# rad/s threshold for flagging sudden head rotation
ANGULAR_VEL_THRESH = 5.0    # rad/s


class SkullRotationDetector:
    """
    For each tracked head:
    1. Compute orientation angle θ(t) from best available keypoint pair:
       Priority: left_ear→right_ear > left_eye→right_eye > nose→midpoint_eyes
    2. Compute angular velocity: ω(t) = (θ(t) - θ(t-1)) × fps
    3. Flag if |ω(t)| > ANGULAR_VEL_THRESH

    Score = min(|ω| / 15.0, 1.0)  — saturates at 15 rad/s
    """

    def __init__(
        self,
        fps:          float = 30.0,
        omega_thresh: float = ANGULAR_VEL_THRESH,
        min_history:  int   = 3,
    ):
        self.fps          = fps
        self.omega_thresh = omega_thresh
        self.min_history  = min_history

        # dict[track_id → deque of angles]
        self._angles: dict[int, deque] = {}

    def _compute_orientation(self, kps: np.ndarray, kp_conf: float = 0.30) -> float | None:
        """
        kps: (5, 3) [x, y, conf] for [nose, l_eye, r_eye, l_ear, r_ear]
        Returns orientation angle in radians, or None if not computable.
        """
        l_ear, r_ear = kps[3], kps[4]
        l_eye, r_eye = kps[1], kps[2]
        nose         = kps[0]

        # Priority 1: ear-to-ear vector (most stable)
        if l_ear[2] > kp_conf and r_ear[2] > kp_conf:
            dx = r_ear[0] - l_ear[0]
            dy = r_ear[1] - l_ear[1]
            return float(np.arctan2(dy, dx))

        # Priority 2: eye-to-eye vector
        if l_eye[2] > kp_conf and r_eye[2] > kp_conf:
            dx = r_eye[0] - l_eye[0]
            dy = r_eye[1] - l_eye[1]
            return float(np.arctan2(dy, dx))

        # Priority 3: nose to midpoint of available eyes
        visible_eyes = [p for p in [l_eye, r_eye] if p[2] > kp_conf]
        if nose[2] > kp_conf and visible_eyes:
            mid = np.mean([e[:2] for e in visible_eyes], axis=0)
            dx  = nose[0] - mid[0]
            dy  = nose[1] - mid[1]
            return float(np.arctan2(dy, dx))

        return None

    def detect(self, states: list[HeadState]) -> list[dict]:
        """
        Returns list of flagged tracks:
            id          : int
            omega_rad_s : float   angular velocity [rad/s]
            score       : float   0–1
        """
        results = []

        for hs in states:
            tid   = hs.track_id
            angle = self._compute_orientation(hs.keypoints)

            if angle is None:
                continue

            if tid not in self._angles:
                self._angles[tid] = deque(maxlen=5)
            self._angles[tid].append(angle)

            if len(self._angles[tid]) < self.min_history:
                continue

            # unwrap to handle ±π discontinuities
            angle_arr    = np.array(self._angles[tid])
            angle_unwrap = np.unwrap(angle_arr)
            # angular velocity: finite difference of last two frames
            omega = abs(angle_unwrap[-1] - angle_unwrap[-2]) * self.fps

            if omega > self.omega_thresh:
                score = min(omega / 15.0, 1.0)
                results.append({
                    "id":          tid,
                    "omega_rad_s": round(omega, 3),
                    "score":       round(score, 4),
                })

        return results
```

---

## File 5 — `impact_buffer.py`

**Purpose:** Merges signals from all 3 stages into impact events. On confirmed
impact, records the event and queues it for HybrIK retrospective processing.

```python
"""
impact_buffer.py — Impact event merger and trigger

Confidence scoring:
    Proximity  : 40% weight  →  proximity_score × 0.40
    Velocity   : 35% weight  →  min(z/10, 1.0)  × 0.35
    Skull rot  : 25% weight  →  skull_score      × 0.25

An event requires at least 2 stages to fire (any single stage alone
can be spurious). Confirmed events are queued for HybrIK processing.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np


CONFIDENCE_THRESHOLD = 0.25   # minimum to record as impact event
MIN_STAGES_REQUIRED  = 2      # at least 2 stages must fire


@dataclass
class ImpactEvent:
    frame_idx:   int
    track_ids:   list[int]       # all track IDs involved
    confidence:  float
    stages:      list[str]       # which stages fired
    details:     dict = field(default_factory=dict)


class ImpactBuffer:
    """
    Accumulates per-frame stage signals and emits ImpactEvents.
    Maintains a cooldown per track pair to avoid duplicate events.
    """

    # frames to suppress re-detection after an event (at 30fps: 1.5s)
    COOLDOWN_FRAMES = 45

    def __init__(
        self,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        min_stages: int = MIN_STAGES_REQUIRED,
    ):
        self.conf_thresh = confidence_threshold
        self.min_stages  = min_stages

        # cooldown tracker: pair_key → last_event_frame
        self._cooldowns: dict[str, int] = {}

        # output: all confirmed events across the whole video
        self.events: list[ImpactEvent] = []

    def _pair_key(self, ids: list[int]) -> str:
        return "-".join(str(i) for i in sorted(ids))

    def process_frame(
        self,
        frame_idx:   int,
        proximity:   list[dict],    # from ProximityDetector
        velocity:    list[dict],    # from KeypointVelocityDetector
        skull_rot:   list[dict],    # from SkullRotationDetector
    ) -> list[ImpactEvent]:
        """
        Merge signals from all 3 stages for one frame.
        Returns list of new ImpactEvents confirmed this frame.
        """
        # index velocity and skull rotation by track ID for fast lookup
        vel_by_id   = {d["id"]: d for d in velocity}
        skull_by_id = {d["id"]: d for d in skull_rot}

        new_events = []

        for prox in proximity:
            id_a  = prox["id_a"]
            id_b  = prox["id_b"]
            p_key = self._pair_key([id_a, id_b])

            # cooldown check
            last_event = self._cooldowns.get(p_key, -self.COOLDOWN_FRAMES - 1)
            if frame_idx - last_event < self.COOLDOWN_FRAMES:
                continue

            stages_fired = ["proximity"]
            conf = prox["score"] * 0.40
            details = {"proximity_dist_norm": prox["dist_norm"],
                       "proximity_score":     prox["score"]}

            # velocity contribution — check EITHER of the two heads
            vel_contrib = 0.0
            for tid in [id_a, id_b]:
                if tid in vel_by_id:
                    z = vel_by_id[tid]["z_score"]
                    contrib = min(z / 10.0, 1.0) * 0.35
                    if contrib > vel_contrib:
                        vel_contrib = contrib
                        details["velocity_z"] = z
                        details["velocity_id"] = tid
            if vel_contrib > 0:
                stages_fired.append("velocity")
                conf += vel_contrib

            # skull rotation contribution — check EITHER head
            skull_contrib = 0.0
            for tid in [id_a, id_b]:
                if tid in skull_by_id:
                    s = skull_by_id[tid]["score"]
                    contrib = s * 0.25
                    if contrib > skull_contrib:
                        skull_contrib = contrib
                        details["skull_omega_rad_s"] = skull_by_id[tid]["omega_rad_s"]
                        details["skull_id"] = tid
            if skull_contrib > 0:
                stages_fired.append("skull_rotation")
                conf += skull_contrib

            if len(stages_fired) < self.min_stages:
                continue
            if conf < self.conf_thresh:
                continue

            event = ImpactEvent(
                frame_idx  = frame_idx,
                track_ids  = [id_a, id_b],
                confidence = round(conf, 4),
                stages     = stages_fired,
                details    = details,
            )
            self._cooldowns[p_key] = frame_idx
            self.events.append(event)
            new_events.append(event)

        return new_events
```

---

## File 6 — `hybrik_retrospective.py`

**Purpose:** Stage 4. Given a confirmed impact event and buffered frames, runs
HybrIK on ±W frames around the impact for each flagged track ID. Returns
R_head(t) rotation matrices for the profiler.

```python
"""
hybrik_retrospective.py — Stage 4 (on-demand)

Runs HybrIK ONLY on buffered frames around a confirmed impact event.
This is the compute-expensive step — but it only ever runs on
~30-90 frames total across the entire video regardless of video length.

HybrIK outputs:
  rot_mats: (1, 24, 3, 3) — rotation matrix per SMPL joint
  SMPL joint 15 = head joint
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
import cv2

HYBRIK_ROOT = Path(__file__).parent.parent / "HybrIK"
sys.path.insert(0, str(HYBRIK_ROOT))

from hybrik.models import builder as hybrik_builder
from hybrik.utils.config import update_config
from easydict import EasyDict as edict

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
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if cfg_file is None:
            cfg_file = str(
                HYBRIK_ROOT / "configs" /
                "256x192_adam_lr1e-3-hrw48_cam_2x_w_pw3d_3dhp.yaml"
            )

        cfg = edict()
        update_config(cfg, cfg_file)
        self.model = hybrik_builder.build_sppe(cfg.MODEL)

        ckpt_path = Path(__file__).parent / ckpt
        state     = torch.load(str(ckpt_path), map_location=device)
        state_dict = state.get("model", state.get("state_dict", state))
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(device).eval()

        self._mean = np.array([0.485, 0.456, 0.406], np.float32)
        self._std  = np.array([0.229, 0.224, 0.225], np.float32)

    def _crop_and_preprocess(
        self,
        frame_bgr: np.ndarray,
        body_box: np.ndarray,   # [x1,y1,x2,y2]
        pad: float = 0.15,
    ) -> torch.Tensor:
        """Crop person from frame, pad, resize to 256×192, normalize."""
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
        crop_res = cv2.resize(crop_rgb, (256, 192))
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

            # rot_mats: (1, 24, 3, 3)
            rot_mats = output.rot_mats.cpu().numpy()
            head_rot = rot_mats[0, HEAD_JOINT_IDX]    # (3, 3)
            rot_by_frame[fidx] = head_rot

        return rot_by_frame
```

---

## File 7 — `brain_injury_profiler.py`

**Purpose:** Final stage. Takes rotation matrices from HybrIKRetrospective and
computes the angular velocity profile + BrIC_R / KLC / DAMAGE scores.

```python
"""
brain_injury_profiler.py

Input : dict[frame_idx → (3,3) rotation matrix] from HybrIKRetrospective
Output: angular velocity time series + BrIC_R, KLC, DAMAGE scores

Key advantage of using rotation matrices (vs position differentiation):
  dR = R(t+1) @ R(t).T         ← exact relative rotation, no noise amplification
  ω(t) = log(dR) / dt          ← axis-angle / Δt = angular velocity [rad/s]
"""

from __future__ import annotations
import numpy as np
from scipy.signal import savgol_filter
from scipy.linalg import logm

# BrIC_R critical value — resultant, direction-independent (Takhounts 2013)
BRIC_OMEGA_CRIT_R = 53.0   # rad/s  → BrIC_R = 1.0 at 50% AIS2+ risk

# DAMAGE model parameters (Gabler 2019)
DAMAGE_OMEGA_N = 30.1      # rad/s  natural frequency
DAMAGE_ZETA    = 0.746     # damping ratio


def _rot_to_rotvec(R: np.ndarray) -> np.ndarray:
    """(3,3) rotation matrix → axis-angle vector (3,)."""
    log_R = logm(R)
    return np.array([log_R[2,1], log_R[0,2], log_R[1,0]], dtype=np.float64)


def compute_omega(
    rot_dict: dict[int, np.ndarray],
    fps: float,
    smooth: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert frame-indexed rotation matrices to angular velocity time series.

    Returns
    -------
    frame_indices : np.ndarray (T-1,)
    omega         : np.ndarray (T-1, 3)  rad/s
    """
    frames_sorted = sorted(rot_dict.keys())
    if len(frames_sorted) < 2:
        return np.array([]), np.zeros((0, 3))

    dt    = 1.0 / fps
    omega = []
    fidxs = []

    for i in range(len(frames_sorted) - 1):
        f1 = frames_sorted[i]
        f2 = frames_sorted[i + 1]
        # handle non-consecutive frames (gaps in buffer)
        actual_dt = (f2 - f1) / fps
        if actual_dt <= 0:
            continue

        R1 = rot_dict[f1]
        R2 = rot_dict[f2]
        dR = R2 @ R1.T
        rvec = _rot_to_rotvec(dR)
        omega.append(rvec / actual_dt)
        fidxs.append(f1)

    if not omega:
        return np.array([]), np.zeros((0, 3))

    omega = np.array(omega)   # (T-1, 3)
    fidxs = np.array(fidxs)

    if smooth and len(omega) >= 7:
        for ax in range(3):
            omega[:, ax] = savgol_filter(omega[:, ax], 7, 2)

    return fidxs, omega


def compute_bric_r(omega: np.ndarray) -> float:
    if len(omega) == 0:
        return 0.0
    return float(np.linalg.norm(omega, axis=1).max()) / BRIC_OMEGA_CRIT_R


def compute_klc_rotation(omega: np.ndarray) -> float:
    """KLC rotation component = peak resultant ω [rad/s]."""
    if len(omega) == 0:
        return 0.0
    return float(np.linalg.norm(omega, axis=1).max())


def compute_damage(omega: np.ndarray, fps: float) -> float:
    """
    DAMAGE: spring-mass convolution driven by ω_magnitude(t).
    Returns peak relative displacement. >0.2 → elevated MPS risk.
    """
    if len(omega) == 0:
        return 0.0
    dt       = 1.0 / fps
    T        = len(omega)
    t_arr    = np.arange(T) * dt
    wn, zeta = DAMAGE_OMEGA_N, DAMAGE_ZETA
    wd       = wn * np.sqrt(max(1.0 - zeta**2, 1e-9))
    h        = np.exp(-zeta * wn * t_arr) * np.sin(wd * t_arr) / wd
    omega_mag = np.linalg.norm(omega, axis=1)
    x_t      = wn**2 * np.convolve(omega_mag, h * dt)[:T]
    return float(np.abs(x_t).max())


def _risk_label(val, elev_thr, high_thr) -> str:
    if val < elev_thr:
        return "LOW"
    elif val < high_thr:
        return "ELEVATED"
    return "HIGH"


class BrainInjuryProfiler:
    """
    Called once per confirmed impact event, after HybrIK retrospective pass.
    """

    def profile(
        self,
        rot_dict: dict[int, np.ndarray],  # frame_idx → (3,3) head rotation matrix
        fps: float,
        track_id: int,
        event_frame: int,
    ) -> dict:
        """
        Returns a full brain injury report for one person in one impact event.
        """
        fidxs, omega = compute_omega(rot_dict, fps)

        if len(omega) == 0:
            return {"track_id": track_id, "error": "insufficient rotation data"}

        omega_peak = float(np.linalg.norm(omega, axis=1).max())
        bric_r     = compute_bric_r(omega)
        klc_rot    = compute_klc_rotation(omega)
        damage     = compute_damage(omega, fps)

        bric_r_risk = _risk_label(bric_r,  0.25, 0.50)
        klc_risk    = _risk_label(klc_rot, 15.0, 30.0)
        damage_risk = _risk_label(damage,  0.10, 0.20)

        risks = [bric_r_risk, klc_risk, damage_risk]
        overall = "HIGH" if "HIGH" in risks else (
                  "ELEVATED" if "ELEVATED" in risks else "LOW")

        return {
            "track_id":         track_id,
            "event_frame":      event_frame,
            "n_frames":         len(rot_dict),
            "frame_indices":    fidxs.tolist(),
            "omega_xyz":        omega.tolist(),       # (T, 3) serializable
            "omega_peak_rad_s": round(omega_peak, 3),
            "bric_r":           round(bric_r,     4),
            "bric_r_risk":      bric_r_risk,
            "klc_rot_rad_s":    round(klc_rot,    3),
            "klc_risk":         klc_risk,
            "damage":           round(damage,      4),
            "damage_risk":      damage_risk,
            "risk_summary":     overall,
        }
```

---

## File 8 — `track_video.py` (complete rewrite)

**Purpose:** Two-pass pipeline orchestrator.

```python
"""
track_video.py — Pipeline entry point

Two-pass processing:
  Pass 1 (fast) : YOLOv8-pose + 3 lightweight stages → find impact timestamps
  Pass 2 (deep) : HybrIK on ±HALF_WINDOW frames per confirmed impact event

Usage:
    python track_video.py --video path/to/video.mp4 [--window 15] [--show]
"""

import argparse
import json
from pathlib import Path

from head_tracker          import HeadKeypointTracker
from proximity_detector    import ProximityDetector
from velocity_detector     import KeypointVelocityDetector
from skull_rotation_detector import SkullRotationDetector
from impact_buffer         import ImpactBuffer, ImpactEvent
from hybrik_retrospective  import HybrIKRetrospective
from brain_injury_profiler import BrainInjuryProfiler


HALF_WINDOW = 15   # frames before and after impact for HybrIK (0.5s at 30fps)


def run(video_path: str, half_window: int = HALF_WINDOW, show: bool = False):

    print(f"\n{'='*60}")
    print(f"NeurivAI v2 — Head Impact Pipeline")
    print(f"Video : {video_path}")
    print(f"{'='*60}\n")

    # ── PASS 1: lightweight detection ─────────────────────────────────────────
    print("[Pass 1] Running head tracking + impact detection...")

    tracker    = HeadKeypointTracker(source=video_path, show=show,
                                     buffer_frames=half_window * 2 + 10)
    proximity  = ProximityDetector()
    velocity   = KeypointVelocityDetector()
    skull_rot  = SkullRotationDetector(fps=30.0)   # updated below
    imp_buf    = ImpactBuffer()

    for frame_idx, frame, head_states in tracker.track():
        # update skull rotation detector fps once we know it
        if frame_idx == 0:
            skull_rot.fps = tracker.fps

        prox_hits   = proximity.detect(head_states)
        vel_hits    = velocity.detect(head_states)
        skull_hits  = skull_rot.detect(head_states)

        new_events  = imp_buf.process_frame(
            frame_idx, prox_hits, vel_hits, skull_hits
        )

        for ev in new_events:
            print(
                f"  [IMPACT DETECTED] frame={ev.frame_idx:05d} | "
                f"tracks={ev.track_ids} | "
                f"conf={ev.confidence:.3f} | "
                f"stages={ev.stages}"
            )

    all_events = imp_buf.events
    print(f"\n[Pass 1 complete] {len(all_events)} impact event(s) found.\n")

    if not all_events:
        print("No impacts detected. Done.")
        return []

    # ── PASS 2: HybrIK retrospective on flagged windows ───────────────────────
    print("[Pass 2] Running HybrIK on impact windows...")

    hybrik   = HybrIKRetrospective()
    profiler = BrainInjuryProfiler()
    reports  = []

    for ev in all_events:
        print(f"\n  Processing event @ frame {ev.frame_idx} "
              f"(tracks {ev.track_ids})...")

        frame_window = tracker.get_frame_window(ev.frame_idx, half_window)
        if not frame_window:
            print("    ⚠ No buffered frames available for this event.")
            continue

        for tid in ev.track_ids:
            rot_dict = hybrik.process_event(frame_window, tid, tracker)

            if not rot_dict:
                print(f"    ⚠ No HybrIK output for track {tid}")
                continue

            report = profiler.profile(
                rot_dict, tracker.fps, tid, ev.frame_idx
            )
            reports.append(report)

            # print summary
            if "error" in report:
                print(f"    track {tid}: {report['error']}")
            else:
                print(
                    f"    track {tid}: "
                    f"ω_peak={report['omega_peak_rad_s']:.1f} rad/s | "
                    f"BrIC_R={report['bric_r']:.3f} ({report['bric_r_risk']}) | "
                    f"KLC={report['klc_rot_rad_s']:.1f} rad/s ({report['klc_risk']}) | "
                    f"DAMAGE={report['damage']:.4f} ({report['damage_risk']}) | "
                    f"RISK={report['risk_summary']}"
                )

    # ── save results ──────────────────────────────────────────────────────────
    out_path = Path(video_path).with_suffix(".impact_report.json")
    # omega_xyz is a list of lists — JSON serializable
    with open(out_path, "w") as f:
        json.dump({"events": [
            {"frame": ev.frame_idx, "tracks": ev.track_ids,
             "confidence": ev.confidence, "stages": ev.stages,
             "details": ev.details}
            for ev in all_events
        ], "profiles": reports}, f, indent=2)

    print(f"\n[Done] Report saved to {out_path}")
    return reports


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",  required=True)
    ap.add_argument("--window", type=int, default=HALF_WINDOW)
    ap.add_argument("--show",   action="store_true")
    args = ap.parse_args()
    run(args.video, args.window, args.show)
```

---

## Final File Structure

```
competition/
├── track_video.py              ← COMPLETE REWRITE (two-pass orchestrator)
├── head_tracker.py             ← NEW  (Stage 0: YOLOv8x-pose + ByteTrack)
├── proximity_detector.py       ← NEW  (Stage 1: normalized head distance)
├── velocity_detector.py        ← NEW  (Stage 2: keypoint centroid z-score)
├── skull_rotation_detector.py  ← NEW  (Stage 3: ear-vector angular velocity)
├── impact_buffer.py            ← NEW  (signal merger + event queue)
├── hybrik_retrospective.py     ← NEW  (Stage 4: on-demand HybrIK)
├── brain_injury_profiler.py    ← NEW  (BrIC_R / KLC / DAMAGE from rot mats)
│
│   ── DELETED ──
├── helmet_tracker.py           ← DELETE (replaced by head_tracker.py)
├── iou_detector.py             ← DELETE (replaced by proximity_detector.py)
├── hot_detector.py             ← DELETE (replaced by skull_rotation_detector.py)
├── impact_detector.py          ← DELETE (replaced by impact_buffer.py)
│
└── models/
    ├── hybrik/
    │   └── hybrik_hrnet.pth    ← download from HybrIK README
    └── hot-c1/                 ← can be deleted

../HybrIK/                      ← git clone + pip install -e .
```

---

## Key Architectural Decisions for Claude Code to Understand

**Why YOLOv8x-pose for Stage 0 instead of HybrIK?**
YOLOv8x-pose runs at 15-30fps on GPU for multi-person scenes. HybrIK runs at
5-10fps per person — at 10 people that is 10× slower than real-time. For the
detection stage we only need 2D keypoints + track IDs. HybrIK's precision is
only needed for the ±0.5s window around a confirmed impact.

**Why skull rotation (Stage 3) replaces HOT?**
HOT detects generic body contact. Skull rotation detects the specific physical
signature of head impact: sudden head snap. It uses the ear-to-ear vector as a
direct 2D readout of head angular velocity in the image plane, computable in
microseconds with no ML. More specific than HOT for this use case.

**Why normalized distance (Stage 1) instead of IoU?**
Keypoints have no area — IoU is undefined. Normalizing by head radius makes the
threshold dimensionless: works for heads at any depth in the scene.

**Why two passes instead of one?**
HybrIK cannot run in real-time for 10+ people. Two passes give you real-time
detection speed (Pass 1) and HybrIK-quality brain injury metrics (Pass 2), with
HybrIK only ever processing the small confirmed windows.

**Why the frame buffer in HeadKeypointTracker?**
Pass 2 needs the raw BGR frames around each impact. The rolling deque keeps only
the last N frames in memory, not the entire video. Size = 2×HALF_WINDOW+10 is
enough to guarantee the window is always available.
