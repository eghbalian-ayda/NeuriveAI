import numpy as np
from collections import defaultdict


class VelocityAnomalyDetector:
    """
    Tracks per-ID centroid displacement across frames.
    A sudden spike in velocity (high z-score vs recent history)
    indicates a physical impact even when boxes don't overlap.
    """

    def __init__(self, window=8, z_thresh=3.0, min_history=5):
        """
        Args:
            window      : number of past frames to keep per track
            z_thresh    : z-score threshold to flag an anomaly
            min_history : minimum positions before anomaly detection starts
                          (prevents false positives on new track entries)
        """
        self.window      = window
        self.z_thresh    = z_thresh
        self.min_history = min_history
        self.history     = defaultdict(list)   # track_id → [(cx, cy), ...]

    @staticmethod
    def _centroid(box):
        return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

    def update(self, detections):
        """
        Args:
            detections: list of {"id": int, "box": [x1,y1,x2,y2], ...}
        Returns:
            list of dicts: {id, velocity, z_score}  — only anomalous tracks
        """
        anomalies = []

        for d in detections:
            tid = d["id"]
            self.history[tid].append(self._centroid(d["box"]))

            # Keep only the last `window` positions
            if len(self.history[tid]) > self.window:
                self.history[tid] = self.history[tid][-self.window:]

            hist = self.history[tid]
            if len(hist) < self.min_history:
                continue

            # Frame-to-frame Euclidean displacement
            velocities = [
                np.hypot(hist[k][0] - hist[k-1][0], hist[k][1] - hist[k-1][1])
                for k in range(1, len(hist))
            ]

            # Compare last velocity to the window's mean/std
            baseline = velocities[:-1]
            mean_v   = float(np.mean(baseline))
            std_v    = float(np.std(baseline)) + 1e-6
            z        = abs(velocities[-1] - mean_v) / std_v

            if z > self.z_thresh:
                anomalies.append({
                    "id":       tid,
                    "velocity": round(velocities[-1], 3),
                    "z_score":  round(z, 3),
                })

        return anomalies

    def reset(self, track_id=None):
        """Clear history for a specific track or all tracks."""
        if track_id is not None:
            self.history.pop(track_id, None)
        else:
            self.history.clear()
