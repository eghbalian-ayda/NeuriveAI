from huggingface_hub import hf_hub_download
from ultralytics import YOLO


class HelmetTracker:
    def __init__(self):
        print("Loading helmet detection model...")
        model_path = hf_hub_download(
            repo_id="keremberke/yolov8m-hard-hat-detection",
            filename="best.pt"
        )
        self.model = YOLO(model_path)

    def track(self, source, show=True, save=True):
        """
        Run helmet tracking on a video source.
        Yields per-frame results with track IDs, boxes, confidences, and class labels.
        """
        results = self.model.track(
            source=source,
            tracker="bytetrack.yaml",
            stream=True,
            show=show,
            save=save,
        )

        for r in results:
            detections = []
            if r.boxes.id is not None:
                track_ids = r.boxes.id.int().cpu().tolist()
                boxes     = r.boxes.xyxy.cpu().tolist()
                confs     = r.boxes.conf.cpu().tolist()
                classes   = r.boxes.cls.int().cpu().tolist()
                names     = r.names
                for tid, box, conf, cls in zip(track_ids, boxes, confs, classes):
                    detections.append({
                        "id":    tid,
                        "label": names[cls],
                        "box":   box,
                        "conf":  conf,
                    })
            yield r.orig_img, detections
