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
