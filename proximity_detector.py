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
            print(f"[PROX] tracks ({a.track_id},{b.track_id}) | "
                  f"size_ratio={max(a.radius_px,b.radius_px)/max(min(a.radius_px,b.radius_px),1.0):.2f} | "
                  f"depth_a={a.depth:.3f} depth_b={b.depth:.3f} "
                  f"depth_diff={abs(a.depth-b.depth):.3f} "
                  f"gate={'PASS' if abs(a.depth-b.depth) <= self.depth_thresh else 'BLOCK'} | "
                  f"dist_norm={np.linalg.norm(a.centroid-b.centroid)/max((a.radius_px+b.radius_px)/2.0,1.0):.2f}")

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
