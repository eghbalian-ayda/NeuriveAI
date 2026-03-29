class IoUDetector:
    """
    Computes pairwise IoU between all tracked boxes each frame.
    A non-zero IoU is a strong (but not necessary) indicator of impact.
    Threshold is kept low so glancing overlaps are also captured.
    """

    def __init__(self, threshold=0.05):
        self.threshold = threshold

    @staticmethod
    def _iou(box_a, box_b):
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter  = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union  = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _proximity(box_a, box_b):
        """Normalised distance between box centres (0 = same centre)."""
        cx_a = (box_a[0] + box_a[2]) / 2
        cy_a = (box_a[1] + box_a[3]) / 2
        cx_b = (box_b[0] + box_b[2]) / 2
        cy_b = (box_b[1] + box_b[3]) / 2
        diag = max(
            ((box_a[2]-box_a[0])**2 + (box_a[3]-box_a[1])**2) ** 0.5,
            ((box_b[2]-box_b[0])**2 + (box_b[3]-box_b[1])**2) ** 0.5,
            1.0,
        )
        dist = ((cx_a - cx_b)**2 + (cy_a - cy_b)**2) ** 0.5
        return dist / diag

    def detect(self, detections):
        """
        Args:
            detections: list of {"id": int, "box": [x1,y1,x2,y2], ...}
        Returns:
            list of dicts: {id_a, id_b, iou, proximity}
            sorted by iou descending.
        """
        results = []
        n = len(detections)
        for i in range(n):
            for j in range(i + 1, n):
                iou = self._iou(detections[i]["box"], detections[j]["box"])
                prox = self._proximity(detections[i]["box"], detections[j]["box"])
                if iou >= self.threshold:
                    results.append({
                        "id_a":      detections[i]["id"],
                        "id_b":      detections[j]["id"],
                        "iou":       round(iou, 4),
                        "proximity": round(prox, 4),
                    })
        return sorted(results, key=lambda x: x["iou"], reverse=True)
