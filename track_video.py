import cv2
from helmet_tracker  import HelmetTracker
from impact_detector import ImpactDetector

VIDEO_PATH  = "/home/ayda/Documents/Githubs/competition/test.mp4"
OUTPUT_PATH = "/home/ayda/Documents/Githubs/competition/output_annotated.mp4"

# ── Init ──────────────────────────────────────────────────────────────
tracker   = HelmetTracker()
impact    = ImpactDetector()
frame_idx = 0

# ── Video writer setup ────────────────────────────────────────────────
cap     = cv2.VideoCapture(VIDEO_PATH)
fps     = cap.get(cv2.CAP_PROP_FPS) or 30
w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()
writer  = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# ── Colours ───────────────────────────────────────────────────────────
GREEN  = (0, 200, 0)
RED    = (0, 0, 220)
YELLOW = (0, 200, 255)
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)

def draw_label(img, text, x, y, bg_color, text_color=WHITE, scale=0.55, thickness=1):
    """Draw a filled-background text label."""
    font   = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + baseline), bg_color, -1)
    cv2.putText(img, text, (x + 2, y), font, scale, text_color, thickness, cv2.LINE_AA)

# ── Main loop ─────────────────────────────────────────────────────────
for frame, detections in tracker.track(VIDEO_PATH, show=False, save=False):
    frame_idx += 1
    events    = [e for e in impact.detect(frame, detections) if e["confidence"] >= 0.18]
    vis       = frame.copy()

    # Build set of impacted IDs for quick lookup
    impacted_ids = set()
    for ev in events:
        sid = str(ev["id"])
        for d in detections:
            if str(d["id"]) in sid.split("-"):
                impacted_ids.add(d["id"])

    # ── Draw helmet boxes ─────────────────────────────────────────────
    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        is_impact = d["id"] in impacted_ids
        color     = RED if is_impact else GREEN
        thick     = 3   if is_impact else 2

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thick)
        draw_label(vis, f"ID{d['id']} {d['conf']:.2f}", x1, y1 - 4, color)

    # ── Draw impact event overlays ────────────────────────────────────
    for i, ev in enumerate(events):
        stages = "+".join(s.upper() for s in ev["stages"])
        label  = f"IMPACT [{stages}]  conf={ev['confidence']:.2f}"
        parts  = ", ".join(ev["parts"]) if ev["parts"] else ""

        # Stack event banners at top-left
        y_base = 28 + i * 44
        draw_label(vis, label, 8, y_base,      RED,    WHITE, scale=0.6, thickness=2)
        if parts:
            draw_label(vis, f"parts: {parts}", 8, y_base + 20, BLACK, YELLOW, scale=0.5)

    # ── Frame counter ─────────────────────────────────────────────────
    draw_label(vis, f"frame {frame_idx}", w - 110, h - 8, BLACK, WHITE, scale=0.5)

    writer.write(vis)

writer.release()
print(f"\nAnnotated video saved to: {OUTPUT_PATH}")
