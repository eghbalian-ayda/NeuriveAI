"""
impact_frame_viz.py

Extracts the impact frame from the source video, overlays the instantaneous
angular velocity vector for each involved head, and saves as JPEG.

Drawn elements per tracked head
--------------------------------
  ── component arrows from head centroid (scaled to head radius) ──
  Red arrow    — ωx  (rotation about image-x axis  →  pitch / nodding)
  Green arrow  — ωy  (rotation about image-y axis  →  yaw  / turning)
  Blue arc     — ωz  (rotation about depth axis    →  roll / tilting)
  Orange arrow — ‖ω‖ resultant (projection of ωx + ωy onto image plane)

  ── labels ──
  Track ID badge, ‖ω‖ magnitude, Δω, risk summary

Usage
-----
    python impact_frame_viz.py                               # uses defaults
    python impact_frame_viz.py --video test.mp4 --report test.impact_report.json --out impact_viz.jpg
"""

from __future__ import annotations
import argparse
import json
import math
import cv2
import numpy as np
from pathlib import Path

from head_tracker import HeadKeypointTracker


# ── colour palette (BGR) ───────────────────────────────────────────────────────
C_X     = ( 50,  50, 220)   # red-ish   — ωx
C_Y     = ( 50, 200,  50)   # green     — ωy
C_Z     = (210,  90,  50)   # blue-ish  — ωz arc
C_MAG   = (  0, 165, 255)   # orange    — ‖ω‖ resultant
C_WHITE = (255, 255, 255)
C_BLACK = (  0,   0,   0)
C_DARK  = ( 12,  22,  35)
C_RED_RISK   = ( 30,  30, 210)
C_ELEV_RISK  = (  0, 165, 255)
C_LOW_RISK   = ( 50, 200,  50)


def _risk_bgr(label: str) -> tuple:
    return {"HIGH": C_RED_RISK, "ELEVATED": C_ELEV_RISK, "LOW": C_LOW_RISK}.get(
        label, C_WHITE)


def _text_box(img, text, x, y, font_scale=0.55, thickness=1,
              fg=C_WHITE, bg=C_DARK):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(img, (x - 3, y - th - 4), (x + tw + 4, y + bl + 2), bg, -1)
    cv2.putText(img, text, (x, y), font, font_scale, fg, thickness, cv2.LINE_AA)


def _arrow(img, cx, cy, dx, dy, color, thick=2, tip=0.22):
    """Draw an arrow if it has meaningful length."""
    ex, ey = int(cx + dx), int(cy + dy)
    length = math.hypot(dx, dy)
    if length < 4:
        return
    cv2.arrowedLine(img, (int(cx), int(cy)), (ex, ey),
                    color, thick, cv2.LINE_AA, tipLength=tip)


def _arc_roll(img, cx, cy, radius, omega_z, color, thick=2):
    """
    Curved arrow arc around the head circle for the ωz (roll) component.
    Positive ωz → clockwise arc (right-hand rule, z into screen).
    """
    if abs(omega_z) < 0.05:
        return
    arc_r = radius + 14
    # arc spans ±50° around the 12 o'clock position
    if omega_z > 0:
        a_start, a_end = -100, -10   # clockwise sweep
    else:
        a_start, a_end = -170, -80  # counter-clockwise sweep

    cv2.ellipse(img, (int(cx), int(cy)), (arc_r, arc_r),
                0, a_start, a_end, color, thick, cv2.LINE_AA)

    # tiny arrowhead at the end of the arc
    tip_rad = math.radians(a_end)
    tip_x   = int(cx + arc_r * math.cos(tip_rad))
    tip_y   = int(cy + arc_r * math.sin(tip_rad))
    tang_ang = tip_rad + math.pi / 2 * (1 if omega_z > 0 else -1)
    cv2.arrowedLine(img,
                    (tip_x, tip_y),
                    (int(tip_x + 10 * math.cos(tang_ang)),
                     int(tip_y + 10 * math.sin(tang_ang))),
                    color, thick, cv2.LINE_AA, tipLength=0.9)


def _draw_axis_legend(img, x0, y0):
    """Small coordinate-axis legend box."""
    pad = 8
    box_w, box_h = 240, 120
    sub = img[y0 - pad: y0 + box_h + pad, x0 - pad: x0 + box_w + pad]
    dark = np.full_like(sub, (18, 28, 42))
    cv2.addWeighted(dark, 0.82, sub, 0.18, 0, sub)
    img[y0 - pad: y0 + box_h + pad, x0 - pad: x0 + box_w + pad] = sub
    cv2.rectangle(img, (x0 - pad, y0 - pad),
                  (x0 + box_w + pad, y0 + box_h + pad), (60, 80, 100), 1)

    cv2.putText(img, "Angular Velocity  w(t_impact)",
                (x0, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                C_WHITE, 1, cv2.LINE_AA)

    items = [
        (C_X,   "arrow",  "wx — pitch (rotation about x-axis)"),
        (C_Y,   "arrow",  "wy — yaw   (rotation about y-axis)"),
        (C_Z,   "arc",    "wz — roll  (rotation about z-axis)"),
        (C_MAG, "arrow",  "||w|| resultant (image-plane projection)"),
    ]
    for i, (col, kind, label) in enumerate(items):
        yi = y0 + 36 + i * 22
        if kind == "arrow":
            cv2.arrowedLine(img, (x0, yi - 4), (x0 + 24, yi - 4),
                            col, 2, cv2.LINE_AA, tipLength=0.4)
        else:  # arc symbol
            cv2.ellipse(img, (x0 + 12, yi - 4), (10, 10),
                        0, -160, -20, col, 2, cv2.LINE_AA)
        cv2.putText(img, label, (x0 + 32, yi),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)


# ── main ──────────────────────────────────────────────────────────────────────

def visualize(video: str, report: str, out: str):

    # ── load report ───────────────────────────────────────────────────────
    with open(report) as f:
        data = json.load(f)

    event       = data["events"][0]
    event_frame = event["frame"]
    profiles    = {p["track_id"]: p
                   for p in data["profiles"]
                   if p["event_frame"] == event_frame}

    print(f"Impact event @ frame {event_frame}  "
          f"tracks {list(profiles.keys())}  conf={event['confidence']:.3f}")

    # ── run tracker up to impact frame (gets real ByteTrack IDs + positions) ─
    print("Running tracker to impact frame...")
    tracker = HeadKeypointTracker(source=video, show=False,
                                  buffer_frames=event_frame + 30)
    target_states: list = []
    target_frame:  np.ndarray | None = None

    for frame_idx, frame, states in tracker.track():
        if frame_idx == event_frame:
            target_states = states
            target_frame  = frame.copy()
            break

    if target_frame is None:
        raise RuntimeError(f"Frame {event_frame} not found in '{video}'.")

    # ── build visualisation canvas ────────────────────────────────────────
    vis = target_frame.copy()
    h, w = vis.shape[:2]

    # subtle dark vignette so overlaid vectors pop
    vignette = np.zeros_like(vis, dtype=np.float32)
    cv2.rectangle(vignette, (0, 0), (w, h), (0, 0, 0), int(min(h, w) * 0.12))
    vis = cv2.addWeighted(vis.astype(np.float32), 0.88,
                          vignette, 0.30, 0).clip(0, 255).astype(np.uint8)

    # ── per-person overlays ───────────────────────────────────────────────
    for hs in target_states:
        tid = hs.track_id
        if tid not in profiles:
            continue

        prof  = profiles[tid]
        fidxs = prof["frame_indices"]   # list of ints
        omegas = prof["omega_xyz"]       # list of [ωx, ωy, ωz]

        # index of the impact frame in the omega time-series
        try:
            fi = fidxs.index(event_frame)
        except ValueError:
            fi = int(np.argmin([abs(f - event_frame) for f in fidxs]))

        omega_vec = np.array(omegas[fi], dtype=float)   # (3,)
        omega_mag = float(np.linalg.norm(omega_vec))

        cx = int(hs.centroid[0])
        cy = int(hs.centroid[1])
        r  = int(hs.radius_px)

        # scale so the *resultant* arrow is 1.6× head radius at peak omega
        peak = max(float(prof["omega_peak_rad_s"]), 1e-3)
        scale = (r * 1.6) / peak

        # ── head circle (impact colour) ───────────────────────────────
        cv2.circle(vis, (cx, cy), r,     C_MAG, 2, cv2.LINE_AA)
        cv2.circle(vis, (cx, cy), r + 1, C_MAG, 1, cv2.LINE_AA)
        cv2.circle(vis, (cx, cy), 4,     C_MAG, -1)

        # ── ωx  — horizontal arrow (right = positive) ─────────────────
        _arrow(vis, cx, cy, omega_vec[0] * scale, 0,  C_X, thick=2)

        # ── ωy  — vertical arrow (down = positive in image coords) ────
        _arrow(vis, cx, cy, 0, omega_vec[1] * scale,  C_Y, thick=2)

        # ── ωz  — roll arc around head circle ─────────────────────────
        _arc_roll(vis, cx, cy, r, omega_vec[2], C_Z, thick=2)

        # ── ‖ω‖ resultant — thick arrow in image-plane direction ───────
        rx = omega_vec[0] * scale
        ry = omega_vec[1] * scale
        _arrow(vis, cx, cy, rx, ry, C_MAG, thick=3, tip=0.20)

        # ── small component endpoint dots ─────────────────────────────
        for dx, dy, col in [(omega_vec[0] * scale, 0, C_X),
                            (0, omega_vec[1] * scale, C_Y)]:
            ex, ey = int(cx + dx), int(cy + dy)
            cv2.circle(vis, (ex, ey), 4, col, -1)

        # ── numeric labels ─────────────────────────────────────────────
        risk     = prof["risk_summary"]
        risk_bgr = _risk_bgr(risk)
        delta    = prof.get("delta_omega_rad_s")
        pulse    = prof.get("pulse_duration_s")

        # track ID above head
        _text_box(vis, f"T{tid}", cx - r, cy - r - 24,
                  font_scale=0.62, thickness=2,
                  fg=C_WHITE, bg=risk_bgr)

        # ‖ω‖ + risk below head
        line1 = f"||w||={omega_mag:.2f} r/s  {risk}"
        _text_box(vis, line1, cx - r, cy + r + 14,
                  font_scale=0.46, thickness=1, fg=C_WHITE, bg=risk_bgr)

        # component readout
        wx, wy, wz = omega_vec
        comp_str = f"wx={wx:+.2f}  wy={wy:+.2f}  wz={wz:+.2f}"
        _text_box(vis, comp_str, cx - r, cy + r + 34,
                  font_scale=0.40, thickness=1,
                  fg=(210, 210, 210), bg=C_DARK)

        # Δω + pulse
        if delta is not None and pulse is not None:
            dp_str = f"Dw={delta:.3f} r/s  FWHM={pulse:.3f}s"
            _text_box(vis, dp_str, cx - r, cy + r + 52,
                      font_scale=0.40, thickness=1,
                      fg=(170, 200, 220), bg=C_DARK)

    # ── title banner ──────────────────────────────────────────────────────
    stages = " + ".join(s.upper() for s in event["stages"])
    title  = f"  IMPACT  frame {event_frame:05d}  |  conf={event['confidence']:.3f}" \
             f"  |  {stages}  "
    _text_box(vis, title, 8, 34,
              font_scale=0.72, thickness=2, fg=C_WHITE, bg=(0, 20, 180))

    # ── legend ────────────────────────────────────────────────────────────
    _draw_axis_legend(vis, 10, h - 155)

    # ── save ──────────────────────────────────────────────────────────────
    cv2.imwrite(out, vis, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Saved: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",  default="test.mp4")
    ap.add_argument("--report", default="test.impact_report.json")
    ap.add_argument("--out",    default="impact_frame_viz.jpg")
    args = ap.parse_args()
    visualize(args.video, args.report, args.out)
