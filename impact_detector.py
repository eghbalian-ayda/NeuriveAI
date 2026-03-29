from hot_detector      import HOTDetector
from iou_detector      import IoUDetector
from velocity_detector import VelocityAnomalyDetector


class ImpactDetector:
    """
    Three-stage impact detection pipeline. Runs every frame.

    Stage 1 – IoU heuristic   : primary real-time trigger (box overlap)
    Stage 2 – HOT model       : confirms contact at body-part level
    Stage 3 – Velocity anomaly: confirms impact via physics

    An impact event is emitted when at least ONE stage fires,
    with a confidence score that increases with each stage agreeing.
    """

    # Per-stage weights for the combined confidence score
    _W_IOU = 0.30
    _W_HOT = 0.50
    _W_VEL = 0.20

    def __init__(
        self,
        hot_enc_path   = "models/hot-c1/encoder_epoch_14.pth",
        hot_dec_path   = "models/hot-c1/decoder_epoch_14.pth",
        iou_threshold  = 0.05,
        hot_threshold  = 0.60,
        vel_z_thresh   = 3.0,
    ):
        self.hot      = HOTDetector(hot_enc_path, hot_dec_path)
        self.iou      = IoUDetector(threshold=iou_threshold)
        self.velocity = VelocityAnomalyDetector(z_thresh=vel_z_thresh)

        self.hot_threshold  = hot_threshold
        self._head_parts    = {"head", "neck", "left_shoulder", "right_shoulder"}

    def detect(self, frame_bgr, detections):
        """
        Args:
            frame_bgr  : current video frame (numpy BGR)
            detections : list of dicts from HelmetTracker
                         {"id", "label", "box", "conf"}
        Returns:
            list of impact events:
            {
                "id"         : track id (or pair "A-B" for IoU),
                "confidence" : float  0–1,
                "stages"     : list of strings showing which stages fired,
                "parts"      : list of contacted body parts (from HOT),
                "details"    : dict with per-stage raw values,
            }
        """
        if not detections:
            return []

        # ── Stage 1: IoU ──────────────────────────────────────────────
        iou_pairs = self.iou.detect(detections)
        iou_hits  = {(p["id_a"], p["id_b"]): p["iou"] for p in iou_pairs}

        # ── Stage 2: HOT ──────────────────────────────────────────────
        hot_results = {}
        for d in detections:
            prob, parts = self.hot.infer(frame_bgr, d["box"])
            hot_results[d["id"]] = {"prob": prob, "parts": parts}

        # ── Stage 3: Velocity anomaly ─────────────────────────────────
        vel_anomalies = self.velocity.update(detections)
        vel_hits      = {a["id"]: a for a in vel_anomalies}

        # ── Combine signals ───────────────────────────────────────────
        events = {}

        # IoU pairs → candidate events
        for (id_a, id_b), iou_val in iou_hits.items():
            key = f"{id_a}-{id_b}"
            events[key] = {
                "id":         key,
                "stages":     ["iou"],
                "confidence": iou_val * self._W_IOU,
                "parts":      [],
                "details":    {"iou": iou_val},
            }

        # HOT results → add to existing events or create solo events
        for d in detections:
            tid  = d["id"]
            hres = hot_results[tid]
            head_parts = [p for p in hres["parts"] if p in self._head_parts]
            if hres["prob"] >= self.hot_threshold and head_parts:
                key = str(tid)
                if key not in events:
                    events[key] = {
                        "id":         tid,
                        "stages":     [],
                        "confidence": 0.0,
                        "parts":      [],
                        "details":    {},
                    }
                events[key]["stages"].append("hot")
                events[key]["parts"]       = head_parts
                events[key]["confidence"] += hres["prob"] * self._W_HOT
                events[key]["details"]["hot_prob"] = hres["prob"]

        # Velocity anomalies → add to existing events or create solo events
        for tid, vinfo in vel_hits.items():
            key = str(tid)
            if key not in events:
                events[key] = {
                    "id":         tid,
                    "stages":     [],
                    "confidence": 0.0,
                    "parts":      [],
                    "details":    {},
                }
            events[key]["stages"].append("velocity")
            # Normalise z-score contribution (cap at z=10 → weight 1.0)
            z_norm = min(vinfo["z_score"] / 10.0, 1.0)
            events[key]["confidence"] += z_norm * self._W_VEL
            events[key]["details"]["velocity"] = vinfo["velocity"]
            events[key]["details"]["z_score"]  = vinfo["z_score"]

        # Clip confidence to [0, 1] and return only events with ≥1 stage
        result = []
        for ev in events.values():
            if ev["stages"]:
                ev["confidence"] = round(min(ev["confidence"], 1.0), 4)
                result.append(ev)

        return sorted(result, key=lambda x: x["confidence"], reverse=True)
