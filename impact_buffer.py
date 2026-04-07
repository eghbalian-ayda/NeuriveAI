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


CONFIDENCE_THRESHOLD  = 0.30   # minimum to record as impact event
MIN_STAGES_REQUIRED   = 2      # at least 2 stages must fire
REQUIRE_VELOCITY      = True   # velocity must be one of the fired stages
                               # prevents proximity+rotation false positives
                               # from players walking past each other


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
        require_velocity: bool = REQUIRE_VELOCITY,
    ):
        self.conf_thresh      = confidence_threshold
        self.min_stages       = min_stages
        self.require_velocity = require_velocity

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
            if self.require_velocity and "velocity" not in stages_fired:
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
