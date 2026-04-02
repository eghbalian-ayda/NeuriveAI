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
        self._centroids:  dict[int, deque] = {}
        self._velocities: dict[int, deque] = {}

    def detect(self, states: list[HeadState]) -> list[dict]:
        """
        Returns list of anomalous tracks:
            id       : int
            velocity : float  [px/frame]
            z_score  : float
        """
        results = []

        for hs in states:
            tid = hs.track_id

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
