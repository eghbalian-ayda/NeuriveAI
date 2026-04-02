"""
impact_moment_viz.py

Reads test_annotated.mp4, plays it normally up to the impact pause,
replaces the freeze section with a clean version:
  - raw frame (no overlays) + 10% zoom centered on the heads
  - neat semi-transparent 3-D Phong sphere per head (unique colour each)
  - one angular velocity resultant arrow inside each sphere
Ends the video there. Saves as a new file.

Usage:
    python impact_moment_viz.py
    python impact_moment_viz.py --annotated test_annotated.mp4 --source test.mp4
                                --report test.impact_report.json
                                --out test_impact_moment.mp4
"""

from __future__ import annotations
import argparse, json, math
import cv2
import numpy as np
from head_tracker import HeadKeypointTracker

# unique sphere colour per head (BGR); vivid, clearly distinct
HEAD_COLORS = [
    (230,  70,  20),   # electric blue  — first track
    ( 20, 150, 255),   # warm orange    — second track
    ( 20, 220, 100),   # lime green
    (200,  30, 200),   # magenta
]

FREEZE_SECS  = 2.0
ZOOM_AMOUNT  = 0.10    # crop this fraction from frame edges (10% zoom-in)
SPHERE_ALPHA = 0.52    # sphere interior transparency (0=invisible, 1=opaque)


# ── 3-D Phong sphere ─────────────────────────────────────────────────────────

def _phong_sphere(img: np.ndarray,
                  cx: int, cy: int, r: int,
                  light: np.ndarray,
                  color_bgr: tuple) -> None:
    """
    Blend a Phong-shaded semi-transparent sphere onto img.

    The specular highlight is driven by `light` (the ω unit vector),
    so the position of the glare encodes all 3 dimensions:
      x, y → the bright spot drifts in the image-plane ω direction
      z    → determines how centred / diffuse the highlight is
    """
    H, W = img.shape[:2]
    y0, y1 = max(0, cy - r), min(H, cy + r + 1)
    x0, x1 = max(0, cx - r), min(W, cx + r + 1)
    if y1 <= y0 or x1 <= x0:
        return

    Y, X = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    Y -= cy; X -= cx

    dist_sq = X ** 2 + Y ** 2
    inside  = dist_sq <= float(r) ** 2

    # surface normal on unit sphere
    z_surf = np.where(inside,
                      np.sqrt(np.maximum(1.0 - dist_sq / (float(r) ** 2), 0.0)),
                      0.0)
    nx = X / r
    ny = Y / r
    nz = z_surf

    # normalise light vector (take positive z so highlight is on front face)
    lx, ly, lz = float(light[0]), float(light[1]), abs(float(light[2]))
    ll = math.sqrt(lx ** 2 + ly ** 2 + lz ** 2)
    if ll > 0:
        lx /= ll; ly /= ll; lz /= ll

    # Phong
    diffuse  = np.maximum(nx * lx + ny * ly + nz * lz, 0.0)
    refl     = 2.0 * nz * lz - lz           # reflection dot view (z-aligned)
    specular = np.maximum(refl, 0.0) ** 14 * 0.9

    intensity = 0.10 + 0.50 * diffuse + specular

    # Fresnel rim — thin bright ring at the edge (glass-like)
    rim   = np.where(inside, np.clip((1.0 - nz) ** 4, 0.0, 1.0) * 0.35, 0.0)
    alpha = np.where(inside,
                     np.clip(intensity * SPHERE_ALPHA + rim, 0.0, 0.85),
                     0.0)

    bc, gc, rc = color_bgr
    patch = img[y0:y1, x0:x1].astype(np.float32)
    for ch, cval in enumerate((bc, gc, rc)):
        # highlight tints toward white; shadow toward the sphere colour
        coloured = cval * (intensity * 0.7 + 0.3) + 255.0 * specular * 0.5
        coloured = np.minimum(coloured, 255.0)
        patch[:, :, ch] = patch[:, :, ch] * (1.0 - alpha) + coloured * alpha
    img[y0:y1, x0:x1] = patch.clip(0, 255).astype(np.uint8)

    # clean circle outline (fully opaque, 1 px AA)
    cv2.circle(img, (cx, cy), r, color_bgr, 1, cv2.LINE_AA)
    # soft outer glow ring for clarity
    cv2.circle(img, (cx, cy), r + 1,
               tuple(min(int(c * 0.55), 255) for c in color_bgr), 1, cv2.LINE_AA)


# ── resultant arrow ───────────────────────────────────────────────────────────

def _resultant_arrow(img: np.ndarray,
                     cx: int, cy: int, r: int,
                     omega_xy_unit: np.ndarray) -> None:
    """White arrow from sphere centre toward the image-plane ω direction."""
    dx, dy = float(omega_xy_unit[0]), float(omega_xy_unit[1])
    if math.hypot(dx, dy) < 1e-6:
        return
    length = r * 0.78        # arrow tip at 78% of sphere radius
    ex = int(cx + dx * length)
    ey = int(cy + dy * length)
    # dark shadow for contrast on any sphere colour
    cv2.arrowedLine(img, (cx, cy), (ex, ey),
                    (10, 10, 10), 5, cv2.LINE_AA, tipLength=0.30)
    # bright white arrow
    cv2.arrowedLine(img, (cx, cy), (ex, ey),
                    (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.30)


# ── main ─────────────────────────────────────────────────────────────────────

def run(annotated: str, source: str, report: str, out: str) -> None:

    # ── load impact data ──────────────────────────────────────────────────
    with open(report) as f:
        data = json.load(f)

    event       = data["events"][0]
    event_frame = event["frame"]
    profiles    = {p["track_id"]: p
                   for p in data["profiles"]
                   if p["event_frame"] == event_frame}

    # ── video metadata ────────────────────────────────────────────────────
    cap = cv2.VideoCapture(source)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 24.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    freeze_n = int(round(fps * FREEZE_SECS))

    # ── get head positions at impact frame ────────────────────────────────
    tracker = HeadKeypointTracker(source=source, show=False,
                                  buffer_frames=event_frame + 30)
    active, raw_frame = [], None
    for fidx, frame, states in tracker.track():
        if fidx == event_frame:
            raw_frame = frame.copy()
            active    = [(hs, profiles[hs.track_id])
                         for hs in states if hs.track_id in profiles]
            break

    if raw_frame is None:
        raise RuntimeError(f"Frame {event_frame} not found in {source!r}.")

    # ── build clean freeze frame ──────────────────────────────────────────
    clean = raw_frame.copy()

    # assign unique colour per track (deterministic sort)
    tid_to_col = {tid: HEAD_COLORS[i % len(HEAD_COLORS)]
                  for i, tid in enumerate(sorted(profiles))}

    for hs, prof in active:
        cx = int(hs.centroid[0])
        cy = int(hs.centroid[1])
        r  = int(hs.radius_px)

        fidxs  = prof["frame_indices"]
        omegas = prof["omega_xyz"]
        try:
            fi = fidxs.index(event_frame)
        except ValueError:
            fi = int(np.argmin([abs(f - event_frame) for f in fidxs]))

        omega     = np.array(omegas[fi], dtype=float)
        omega_mag = float(np.linalg.norm(omega))
        if omega_mag < 1e-9:
            continue
        omega_unit = omega / omega_mag

        _phong_sphere(clean, cx, cy, r, omega_unit, tid_to_col[hs.track_id])
        _resultant_arrow(clean, cx, cy, r, omega_unit[:2])

    # ── 10% zoom centered on head midpoint ───────────────────────────────
    if active:
        mid_x = int(np.mean([hs.centroid[0] for hs, _ in active]))
        mid_y = int(np.mean([hs.centroid[1] for hs, _ in active]))
    else:
        mid_x, mid_y = src_w // 2, src_h // 2

    cw = int(src_w * (1.0 - ZOOM_AMOUNT))
    ch = int(src_h * (1.0 - ZOOM_AMOUNT))
    x0 = int(np.clip(mid_x - cw // 2, 0, src_w - cw))
    y0 = int(np.clip(mid_y - ch // 2, 0, src_h - ch))

    freeze_vis = cv2.resize(clean[y0:y0 + ch, x0:x0 + cw],
                            (src_w, src_h),
                            interpolation=cv2.INTER_LANCZOS4)

    # ── write output video ────────────────────────────────────────────────
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (src_w, src_h))

    # read annotated video up to (not including) the impact freeze section
    ann     = cv2.VideoCapture(annotated)
    written = 0
    while True:
        ret, frame = ann.read()
        if not ret or written >= event_frame:
            break
        writer.write(frame)
        written += 1
    ann.release()

    # freeze on the clean viz frame
    for _ in range(freeze_n):
        writer.write(freeze_vis)

    writer.release()
    print(f"Saved → {out}  ({written} normal frames + {freeze_n} freeze frames)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotated", default="test_annotated.mp4")
    ap.add_argument("--source",   default="test.mp4")
    ap.add_argument("--report",   default="test.impact_report.json")
    ap.add_argument("--out",      default="test_impact_moment.mp4")
    args = ap.parse_args()
    run(args.annotated, args.source, args.report, args.out)
