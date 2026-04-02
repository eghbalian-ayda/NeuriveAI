"""
impact_frame_viz.py

Extracts the impact frame, draws a clean angular velocity overlay per head:
  • Fade sphere  — 3-D glow whose centre shifts toward (ωx, ωy) and whose
                   depth (ωz) modulates a secondary inner halo; encodes all
                   three vector dimensions without text or axes
  • Resultant arrow — single sharp arrow showing ‖ω‖ image-plane projection

Output is cropped to the head region and up-scaled back to the original
video resolution (zoom with no resolution loss).

Usage:
    python impact_frame_viz.py
    python impact_frame_viz.py --video test.mp4 --report test.impact_report.json --out impact_frame_viz.jpg
"""

from __future__ import annotations
import argparse, json, math
import cv2
import numpy as np
from head_tracker import HeadKeypointTracker


# ── tunables ──────────────────────────────────────────────────────────────────
SPHERE_COLOR  = (255, 220, 120)   # BGR warm-white / amber  (sphere glow)
ARROW_COLOR   = (0,   200, 255)   # BGR vivid cyan-orange   (resultant arrow)
ARROW_THICK   = 5
SPHERE_RADIUS_FACTOR = 2.5        # sphere = head_radius × this
SPHERE_ALPHA_MAX     = 0.72       # peak glow opacity
ARROW_SCALE_FACTOR   = 1.8        # arrow reaches factor×head_radius at peak ω
ZOOM_PAD_FACTOR      = 1.6        # padding around head bounding box (× head r)


# ── sphere ────────────────────────────────────────────────────────────────────

def _fade_sphere(img: np.ndarray,
                 cx: int, cy: int, r: int,
                 omega: np.ndarray,
                 peak: float,
                 color_bgr: tuple) -> None:
    """
    Blend a 2-D Gaussian glow onto img.

    Encodes all three ω dimensions:
      x, y  → the glow's bright-spot is shifted in the (ωx, ωy) direction
      z     → a secondary inner halo brightens when ωz is large (depth spin)
    """
    H, W = img.shape[:2]
    wx, wy, wz = float(omega[0]), float(omega[1]), float(omega[2])
    mag  = math.sqrt(wx**2 + wy**2 + wz**2)
    norm = mag / max(peak, 1e-3)

    sr = int(r * SPHERE_RADIUS_FACTOR)

    y0, y1 = max(0, cy - sr), min(H, cy + sr + 1)
    x0, x1 = max(0, cx - sr), min(W, cx + sr + 1)
    Y, X = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    Y -= cy
    X -= cx

    # ── outer glow: Gaussian shifted toward (ωx, ωy) ──────────────────────
    shift     = sr * 0.32
    sx        = (wx / max(peak, 1e-3)) * shift
    sy        = (wy / max(peak, 1e-3)) * shift
    sigma_out = sr / 2.2

    dist_sq   = (X - sx)**2 + (Y - sy)**2
    glow_out  = np.exp(-dist_sq / (2 * sigma_out**2))

    # smooth circular fade at boundary
    edge      = np.sqrt(X**2 + Y**2)
    edge_fade = np.clip(1.0 - (edge / sr)**2.5, 0.0, 1.0)
    glow_out *= edge_fade

    # ── inner halo: tighter Gaussian, brightened by |ωz| (depth spin) ─────
    wz_norm   = np.clip(abs(wz) / max(peak, 1e-3), 0.0, 1.0)
    sigma_in  = sr / 5.0
    glow_in   = np.exp(-(X**2 + Y**2) / (2 * sigma_in**2)) * wz_norm * 0.6

    glow = np.clip(glow_out + glow_in, 0.0, 1.0)

    # final alpha
    alpha = (glow * SPHERE_ALPHA_MAX * norm).clip(0.0, SPHERE_ALPHA_MAX)

    # blend colour into image patch
    bc, gc, rc = color_bgr
    patch = img[y0:y1, x0:x1].astype(np.float32)
    for ch, cval in enumerate((bc, gc, rc)):
        patch[:, :, ch] = patch[:, :, ch] * (1.0 - alpha) + cval * alpha
    img[y0:y1, x0:x1] = patch.clip(0, 255).astype(np.uint8)


# ── resultant arrow ───────────────────────────────────────────────────────────

def _resultant_arrow(img: np.ndarray,
                     cx: int, cy: int, r: int,
                     omega: np.ndarray,
                     peak: float) -> None:
    """Sharp coloured arrow for the image-plane projection of ‖ω‖."""
    wx, wy = float(omega[0]), float(omega[1])
    scale  = (r * ARROW_SCALE_FACTOR) / max(peak, 1e-3)
    dx, dy = wx * scale, wy * scale

    if math.hypot(dx, dy) < 6:
        return

    ex, ey = int(cx + dx), int(cy + dy)
    # dark drop-shadow for contrast on any background
    cv2.arrowedLine(img, (cx, cy), (ex, ey),
                    (5, 5, 5), ARROW_THICK + 4, cv2.LINE_AA, tipLength=0.24)
    # main arrow
    cv2.arrowedLine(img, (cx, cy), (ex, ey),
                    ARROW_COLOR, ARROW_THICK, cv2.LINE_AA, tipLength=0.24)
    # bright core highlight
    cv2.arrowedLine(img, (cx, cy), (ex, ey),
                    (255, 255, 255), max(ARROW_THICK - 3, 1), cv2.LINE_AA,
                    tipLength=0.24)


# ── main ──────────────────────────────────────────────────────────────────────

def visualize(video: str, report: str, out: str) -> None:

    # load report
    with open(report) as f:
        data = json.load(f)

    event       = data["events"][0]
    event_frame = event["frame"]
    profiles    = {p["track_id"]: p
                   for p in data["profiles"]
                   if p["event_frame"] == event_frame}

    # run tracker to get real ByteTrack positions at the impact frame
    tracker = HeadKeypointTracker(source=video, show=False,
                                  buffer_frames=event_frame + 30)
    target_states, target_frame = [], None
    for fidx, frame, states in tracker.track():
        if fidx == event_frame:
            target_states = states
            target_frame  = frame.copy()
            break

    if target_frame is None:
        raise RuntimeError(f"Frame {event_frame} not found in {video!r}.")

    src_h, src_w = target_frame.shape[:2]
    vis = target_frame.copy()

    # collect heads that have profiles
    active = [(hs, profiles[hs.track_id])
              for hs in target_states if hs.track_id in profiles]

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

        omega = np.array(omegas[fi], dtype=float)
        peak  = float(prof["omega_peak_rad_s"])

        _fade_sphere(vis, cx, cy, r, omega, peak, SPHERE_COLOR)
        _resultant_arrow(vis, cx, cy, r, omega, peak)

    # ── zoom: crop to head bounding box, resize back to source resolution ──
    if active:
        pad = ZOOM_PAD_FACTOR
        xs, ys, rs = zip(*[(int(hs.centroid[0]), int(hs.centroid[1]),
                             int(hs.radius_px)) for hs, _ in active])
        margin = int(max(rs) * pad)
        x0 = max(0,     min(xs) - margin)
        x1 = min(src_w, max(xs) + margin)
        y0 = max(0,     min(ys) - margin)
        y1 = min(src_h, max(ys) + margin)
        cropped = vis[y0:y1, x0:x1]
        vis = cv2.resize(cropped, (src_w, src_h), interpolation=cv2.INTER_LANCZOS4)

    cv2.imwrite(out, vis, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Saved: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",  default="test.mp4")
    ap.add_argument("--report", default="test.impact_report.json")
    ap.add_argument("--out",    default="impact_frame_viz.jpg")
    args = ap.parse_args()
    visualize(args.video, args.report, args.out)
