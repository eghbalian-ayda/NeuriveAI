"""
impact_moment_viz.py

Reads test_annotated.mp4, plays normally up to the impact freeze, then
replaces it with a 3-second cinematic sequence:

  Phase 1  (1 s)  — hold on the clean raw frame, no zoom, no overlays
  Phase 2  (1 s)  — smooth ease-in-out zoom toward the heads
  Phase 3  (1 s)  — spheres + vectors fade in at full zoom

Sphere style: Blinn-Phong shaded, no hard edge, fully edgeless — the
boundary dissolves with a smooth alpha falloff so no circle outline is visible.
The specular highlight is driven by the ω unit vector, encoding all three
dimensions in the glare position.

Usage:
    python impact_moment_viz.py
    python impact_moment_viz.py --annotated test_annotated.mp4
                                --source test.mp4
                                --report test.impact_report.json
                                --out test_impact_moment.mp4
"""

from __future__ import annotations
import argparse, json, math
import cv2
import numpy as np
from head_tracker import HeadKeypointTracker

HEAD_COLORS = [
    (230,  60,  20),   # electric blue
    ( 15, 145, 255),   # warm orange
    ( 20, 220, 100),   # lime
    (200,  30, 200),   # magenta
]

ZOOM_AMOUNT  = 0.10    # final crop fraction (10 % zoom-in)
SPHERE_ALPHA = 0.58    # peak interior opacity


# ── easing ────────────────────────────────────────────────────────────────────

def _ease(t: float) -> float:
    """Cubic ease-in-out, t ∈ [0, 1]."""
    return t * t * (3.0 - 2.0 * t)


# ── zoom helper ───────────────────────────────────────────────────────────────

def _zoom_frame(base: np.ndarray, progress: float,
                mid_x: int, mid_y: int,
                src_w: int, src_h: int) -> np.ndarray:
    """Return base frame cropped/upscaled by progress ∈ [0, 1]."""
    if progress <= 0.0:
        return base
    z  = ZOOM_AMOUNT * progress
    cw = int(src_w * (1.0 - z))
    ch = int(src_h * (1.0 - z))
    x0 = int(np.clip(mid_x - cw // 2, 0, src_w - cw))
    y0 = int(np.clip(mid_y - ch // 2, 0, src_h - ch))
    return cv2.resize(base[y0:y0 + ch, x0:x0 + cw],
                      (src_w, src_h), interpolation=cv2.INTER_LANCZOS4)


# ── 3-D Blinn-Phong sphere, no hard edge ─────────────────────────────────────

def _phong_sphere(img: np.ndarray,
                  cx: int, cy: int, r: int,
                  light: np.ndarray,
                  color_bgr: tuple,
                  global_alpha: float = 1.0) -> None:
    """
    Edgeless 3-D sphere via Blinn-Phong shading.

    Boundary: alpha fades to zero over the outer ~8 % of the radius so no
    circle outline is visible — the sphere dissolves into the background.

    Specular highlight position = ω unit vector direction → encodes x, y, z.
    """
    if global_alpha <= 0.0:
        return

    H, W = img.shape[:2]
    # extend patch by a few pixels so the soft edge isn't clipped
    pad = max(6, int(r * 0.10))
    y0, y1 = max(0, cy - r - pad), min(H, cy + r + pad + 1)
    x0, x1 = max(0, cx - r - pad), min(W, cx + r + pad + 1)
    if y1 <= y0 or x1 <= x0:
        return

    Y, X = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    Y -= cy; X -= cx

    dist    = np.sqrt(X ** 2 + Y ** 2)
    rf      = float(r)

    # ── surface normal (defined everywhere; zero outside sphere) ───────────
    z2      = np.maximum(rf ** 2 - X ** 2 - Y ** 2, 0.0)
    nz      = np.sqrt(z2) / rf          # 0 at edge, 1 at centre
    nx      = X / rf
    ny      = Y / rf

    # ── normalise light = ω unit vector ────────────────────────────────────
    lx, ly  = float(light[0]), float(light[1])
    lz      = abs(float(light[2]))      # always face the viewer
    ll      = math.sqrt(lx ** 2 + ly ** 2 + lz ** 2)
    if ll > 0:
        lx /= ll; ly /= ll; lz /= ll

    # ── Blinn-Phong ────────────────────────────────────────────────────────
    # view direction V = (0, 0, 1), halfway vector H = normalize(L + V)
    hx, hy, hz = lx, ly, lz + 1.0
    hl = math.sqrt(hx ** 2 + hy ** 2 + hz ** 2)
    hx /= hl; hy /= hl; hz /= hl

    diffuse  = np.maximum(nx * lx + ny * ly + nz * lz, 0.0)
    specular = np.maximum(nx * hx + ny * hy + nz * hz, 0.0) ** 22 * 1.0

    intensity = 0.08 + 0.48 * diffuse + specular

    # ── edgeless boundary: smooth falloff over outer 8 % of radius ─────────
    #    inside sphere: (rf - dist) / soft_band in [0, 1]
    #    outside sphere: 0
    soft_band = max(rf * 0.08, 4.0)
    boundary  = np.clip((rf - dist) / soft_band, 0.0, 1.0)
    # extra cubic smoothing so it truly melts away
    boundary  = boundary * boundary * (3.0 - 2.0 * boundary)

    # ── per-pixel alpha ─────────────────────────────────────────────────────
    alpha = np.clip(intensity * SPHERE_ALPHA, 0.0, SPHERE_ALPHA) \
            * boundary * global_alpha

    # ── colour: sphere colour + specular white highlight ───────────────────
    bc, gc, rc = color_bgr
    patch = img[y0:y1, x0:x1].astype(np.float32)
    for ch_i, cval in enumerate((bc, gc, rc)):
        lit    = np.minimum(cval * (intensity * 0.75 + 0.25)
                            + 255.0 * specular * 0.55, 255.0)
        patch[:, :, ch_i] = patch[:, :, ch_i] * (1.0 - alpha) + lit * alpha
    img[y0:y1, x0:x1] = patch.clip(0, 255).astype(np.uint8)


# ── resultant arrow (fade-able) ───────────────────────────────────────────────

_ARROW_COLOR = (0, 255, 60)   # electric lime — vibrant against any sphere colour


def _resultant_arrow(img: np.ndarray,
                     cx: int, cy: int, r: int,
                     omega_xy_unit: np.ndarray,
                     global_alpha: float = 1.0) -> None:
    if global_alpha <= 0.0:
        return
    dx, dy = float(omega_xy_unit[0]), float(omega_xy_unit[1])
    if math.hypot(dx, dy) < 1e-6:
        return

    length = r * 0.76
    ex = int(cx + dx * length)
    ey = int(cy + dy * length)

    overlay = img.copy()
    # thin dark shadow for contrast (2 px)
    cv2.arrowedLine(overlay, (cx, cy), (ex, ey),
                    (5, 5, 5), 3, cv2.LINE_AA, tipLength=0.30)
    # sharp 1-px vibrant arrow
    cv2.arrowedLine(overlay, (cx, cy), (ex, ey),
                    _ARROW_COLOR, 1, cv2.LINE_AA, tipLength=0.30)
    cv2.addWeighted(overlay, global_alpha, img, 1.0 - global_alpha, 0, img)


# ── main ─────────────────────────────────────────────────────────────────────

def run(annotated: str, source: str, report: str, out: str) -> None:

    with open(report) as f:
        data = json.load(f)

    event       = data["events"][0]
    event_frame = event["frame"]
    profiles    = {p["track_id"]: p
                   for p in data["profiles"]
                   if p["event_frame"] == event_frame}

    # video metadata
    cap   = cv2.VideoCapture(source)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 24.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    phase_n = int(round(fps))          # frames per phase (1 s each)

    # ── get head states + raw frame at impact ─────────────────────────────
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
        raise RuntimeError(f"Frame {event_frame} not found.")

    # zoom centre = midpoint of all involved heads
    if active:
        mid_x = int(np.mean([hs.centroid[0] for hs, _ in active]))
        mid_y = int(np.mean([hs.centroid[1] for hs, _ in active]))
    else:
        mid_x, mid_y = src_w // 2, src_h // 2

    # pre-compute the fully-zoomed clean base (no spheres)
    clean_zoomed = _zoom_frame(raw_frame, 1.0, mid_x, mid_y, src_w, src_h)

    # per-track data
    tid_to_col = {tid: HEAD_COLORS[i % len(HEAD_COLORS)]
                  for i, tid in enumerate(sorted(profiles))}

    head_data = []
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
        head_data.append({
            "cx": cx, "cy": cy, "r": r,
            "unit": omega / omega_mag,
            "color": tid_to_col[hs.track_id],
        })

    def _draw_spheres_arrows(base: np.ndarray, alpha: float) -> np.ndarray:
        """Return a copy of base with spheres+arrows at opacity alpha."""
        img = base.copy()
        for hd in head_data:
            _phong_sphere(img, hd["cx"], hd["cy"], hd["r"],
                          hd["unit"], hd["color"], global_alpha=alpha)
            _resultant_arrow(img, hd["cx"], hd["cy"], hd["r"],
                             hd["unit"][:2], global_alpha=alpha)
        return img

    # ── write output video ────────────────────────────────────────────────
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (src_w, src_h))

    # — pass-through: normal annotated frames up to the impact —
    ann     = cv2.VideoCapture(annotated)
    written = 0
    while True:
        ret, frame = ann.read()
        if not ret or written >= event_frame:
            break
        writer.write(frame)
        written += 1
    ann.release()

    # — phase 1: hold on raw clean frame (no zoom, no spheres) —
    for _ in range(phase_n):
        writer.write(raw_frame)

    # — phase 2: smooth zoom, no spheres —
    for i in range(phase_n):
        t       = i / max(phase_n - 1, 1)
        progress = _ease(t)
        writer.write(_zoom_frame(raw_frame, progress, mid_x, mid_y, src_w, src_h))

    # — phase 3: spheres + arrows fade in on fully-zoomed frame —
    for i in range(phase_n):
        t     = i / max(phase_n - 1, 1)
        alpha = _ease(t)
        writer.write(_draw_spheres_arrows(clean_zoomed, alpha))

    writer.release()
    total = written + phase_n * 3
    print(f"Saved → {out}  "
          f"({written} normal + {phase_n}+{phase_n}+{phase_n} cinematic  "
          f"= {total} frames  {total/fps:.1f} s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotated", default="test_annotated.mp4")
    ap.add_argument("--source",   default="test.mp4")
    ap.add_argument("--report",   default="test.impact_report.json")
    ap.add_argument("--out",      default="test_impact_moment.mp4")
    args = ap.parse_args()
    run(args.annotated, args.source, args.report, args.out)
