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
