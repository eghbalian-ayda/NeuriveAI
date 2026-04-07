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
                # imgsz=1920,
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
