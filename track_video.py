"""
track_video.py — Unified pipeline entry point

Three-pass processing:
  Pass 1: YOLOv8x-pose + Depth Anything V2 + 3 lightweight detectors
          → depth-gated impact detection → impact timestamps
  Pass 2: HybrIK retrospective on ±HALF_WINDOW frames
          → rotation matrices → BrIC_R / KLC / DAMAGE
  Pass 3: Wu et al. CNN → regional MPS → TBI probability + heatmap PNG
          (saved as impact_XXXXX_track_N_tbi.png — not overlaid on video)

Outputs:
  *_annotated.mp4          — annotated video (head circles, badges, freeze)
  *.impact_report.json     — full metrics including TBI probability
  impact_*_tbi.png         — TBI heatmap + bar chart per impact per track

Usage:
    python track_video.py --video path/to/video.mp4 [--window 15] [--show]
    python track_video.py --video path/to/video.mp4 --no-depth
"""

import argparse
import json
import cv2
import numpy as np
from pathlib import Path

from depth_estimator         import DepthEstimator
from head_tracker            import HeadKeypointTracker, HeadState, KP_CONF_THRESH
from proximity_detector      import ProximityDetector
from velocity_detector       import KeypointVelocityDetector
from skull_rotation_detector import SkullRotationDetector
from impact_buffer           import ImpactBuffer, ImpactEvent
from hybrik_retrospective    import HybrIKRetrospective
from brain_injury_profiler   import BrainInjuryProfiler
from strain_estimator        import StrainEstimator
from tbi_visualizer          import TBIVisualizer


HALF_WINDOW = 15    # ±frames around impact for HybrIK (0.5s at 30fps)
FREEZE_SECS = 2.0   # seconds to hold on each impact frame in the output video


# ── colours ───────────────────────────────────────────────────────────────────
C_GREEN  = (0,   210,  60)
C_RED    = (0,   30,  220)
C_YELLOW = (0,   220, 255)
C_CYAN   = (220, 220,   0)
C_WHITE  = (255, 255, 255)
C_BLACK  = (0,     0,   0)
C_ORANGE = (0,   140, 255)

KP_COLORS = [C_WHITE, C_CYAN, C_CYAN, C_YELLOW, C_YELLOW]  # nose, eyes, ears


def _label(img, text, x, y, bg, fg=C_WHITE, scale=0.55, thick=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 6, y + bl + 2), bg, -1)
    cv2.putText(img, text, (x + 3, y), font, scale, fg, thick, cv2.LINE_AA)


def _draw_frame(
    frame:             np.ndarray,
    head_states:       list[HeadState],
    impact_ids:        set[int],
    frame_idx:         int,
    events_this_frame: list[ImpactEvent],
    reports_by_tid:    dict[int, dict],
) -> np.ndarray:
    """Render one annotated frame."""
    vis = frame.copy()

    # ── red vignette overlay on impact frames ─────────────────────────────
    if impact_ids:
        overlay = vis.copy()
        h, w = vis.shape[:2]
        border = int(min(h, w) * 0.06)
        cv2.rectangle(overlay, (0, 0), (w, h), C_RED, border)
        cv2.addWeighted(overlay, 0.55, vis, 0.45, 0, vis)

    # ── per-person head visualisation ─────────────────────────────────────
    for hs in head_states:
        is_impact = hs.track_id in impact_ids
        color     = C_RED if is_impact else C_GREEN
        cx, cy    = int(hs.centroid[0]), int(hs.centroid[1])
        r         = int(hs.radius_px)

        cv2.circle(vis, (cx, cy), r, color, 3 if is_impact else 2)
        cv2.circle(vis, (cx, cy), 3, color, -1)

        for kp, kp_col in zip(hs.keypoints, KP_COLORS):
            if kp[2] > KP_CONF_THRESH:
                cv2.circle(vis, (int(kp[0]), int(kp[1])), 4, kp_col, -1)

        id_text = f"T{hs.track_id}"
        _label(vis, id_text, cx - r, cy - r - 18,
               bg=color, fg=C_WHITE, scale=0.55, thick=2 if is_impact else 1)

        # brain injury badge — ω, BrIC, risk only (no TBI % on video)
        rep = reports_by_tid.get(hs.track_id)
        if is_impact and rep and "error" not in rep:
            risk  = rep["risk_summary"]
            omega = rep["omega_peak_rad_s"]
            bric  = rep["bric_r"]
            badge = f"w={omega:.1f}r/s  BrIC={bric:.2f}  {risk}"
            b_col = C_RED if risk == "HIGH" else C_ORANGE if risk == "ELEVATED" else C_GREEN
            _label(vis, badge, cx - r, cy + r + 14,
                   bg=b_col, fg=C_WHITE, scale=0.45, thick=1)

    # ── impact event banners at top ───────────────────────────────────────
    for i, ev in enumerate(events_this_frame):
        stages = "+".join(s.upper()[:3] for s in ev.stages)
        line1  = f"  IMPACT  tracks {ev.track_ids}  conf={ev.confidence:.2f}  [{stages}]  "
        y_base = 38 + i * 52
        _label(vis, line1, 8, y_base, bg=C_RED, fg=C_WHITE, scale=0.75, thick=2)

    # ── frame counter ─────────────────────────────────────────────────────
    h, w = vis.shape[:2]
    _label(vis, f"frame {frame_idx:05d}", w - 130, h - 10,
           bg=C_BLACK, fg=C_WHITE, scale=0.48)

    return vis


# ── main ──────────────────────────────────────────────────────────────────────

def run(
    video_path: str,
    half_window: int  = HALF_WINDOW,
    show:        bool = False,
    use_depth:   bool = True,
):
    print(f"\n{'='*60}")
    print(f"NeurivAI — Complete Head Impact Pipeline")
    print(f"Video      : {video_path}")
    print(f"Depth gate : {'ON' if use_depth else 'OFF'}")
    print(f"{'='*60}\n")

    out_dir = str(Path(video_path).parent)

    # read video metadata up front (needed for writer + freeze calculation)
    cap0 = cv2.VideoCapture(video_path)
    total_frames = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT)) or 9999
    src_fps      = cap0.get(cv2.CAP_PROP_FPS) or 30.0
    src_w        = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h        = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap0.release()

    freeze_frames = int(round(src_fps * FREEZE_SECS))

    # ── initialise all modules ─────────────────────────────────────────────

    depth_est = None
    if use_depth:
        try:
            depth_est = DepthEstimator()
            print("[Init] Depth Anything V2 Small loaded\n")
        except (FileNotFoundError, ModuleNotFoundError) as e:
            print(f"[Init] WARNING: {e}")
            print("[Init] Continuing without depth gate — "
                  "proximity may have false positives\n")

    tracker   = HeadKeypointTracker(
        source=video_path,
        show=show,
        buffer_frames=total_frames + 10,   # buffer all frames for render pass
    )
    # size ratio gate only meaningful when depth is active —
    # without depth, two heads at different apparent scales can still collide
    proximity  = ProximityDetector(
        size_ratio_max = 1.8 if depth_est is not None else float("inf")
    )
    velocity   = KeypointVelocityDetector()
    skull_rot  = SkullRotationDetector(fps=src_fps)
    imp_buf    = ImpactBuffer()

    hybrik     = HybrIKRetrospective()
    profiler   = BrainInjuryProfiler()
    strain_est = StrainEstimator()
    tbi_viz    = TBIVisualizer(output_dir=out_dir)

    # ── PASS 1: depth-gated lightweight detection ──────────────────────────
    print("[Pass 1] Head tracking + depth-gated impact detection...")

    all_frame_states: dict[int, list[HeadState]] = {}

    for frame_idx, frame, head_states in tracker.track(depth_estimator=depth_est):
        if frame_idx == 0:
            skull_rot.fps = tracker.fps

        prox_hits  = proximity.detect(head_states)
        vel_hits   = velocity.detect(head_states)
        skull_hits = skull_rot.detect(head_states)

        new_events = imp_buf.process_frame(
            frame_idx, prox_hits, vel_hits, skull_hits
        )
        all_frame_states[frame_idx] = head_states

        for ev in new_events:
            print(
                f"  [IMPACT] frame={ev.frame_idx:05d} | "
                f"tracks={ev.track_ids} | conf={ev.confidence:.3f} | "
                f"stages={ev.stages}"
            )

    all_events = imp_buf.events
    print(f"\n[Pass 1 complete] {len(all_events)} impact event(s) found.\n")

    # ── PASSES 2 + 3: HybrIK → brain injury → TBI estimation ─────────────
    reports_by_event: dict[int, dict[int, dict]] = {}   # frame_idx → tid → report

    if all_events:
        print("[Pass 2+3] HybrIK + brain injury + TBI estimation...")

        for ev in all_events:
            print(f"\n  Event @ frame {ev.frame_idx} — tracks {ev.track_ids}")

            frame_window = tracker.get_frame_window(ev.frame_idx, half_window)
            if not frame_window:
                print("    ⚠ No buffered frames — event too early in video")
                continue

            reports_by_event[ev.frame_idx] = {}

            for tid in ev.track_ids:
                rot_dict = hybrik.process_event(frame_window, tid, tracker)
                if not rot_dict:
                    print(f"    ⚠ track {tid}: no HybrIK output")
                    continue

                report = profiler.profile(rot_dict, tracker.fps, tid, ev.frame_idx)
                reports_by_event[ev.frame_idx][tid] = report

                if "error" in report:
                    print(f"    track {tid}: {report['error']}")
                    continue

                print(
                    f"    track {tid}: "
                    f"ω_peak={report['omega_peak_rad_s']:.1f} rad/s | "
                    f"BrIC_R={report['bric_r']:.3f} ({report['bric_r_risk']}) | "
                    f"KLC={report['klc_rot_rad_s']:.1f} rad/s ({report['klc_risk']}) | "
                    f"DAMAGE={report['damage']:.4f} ({report['damage_risk']}) | "
                    f"RISK={report['risk_summary']}"
                )

                # ── Pass 3: ω(t) → MPS → TBI probability + heatmap PNG ────
                omega_np = np.array(report["omega_xyz"])

                regional_mps = strain_est.estimate(
                    omega        = omega_np,
                    damage_score = report.get("damage", 0.0),
                )

                tbi_result = tbi_viz.visualize(
                    regional_mps = regional_mps,
                    track_id     = tid,
                    event_frame  = ev.frame_idx,
                )

                report["tbi_probability_pct"] = tbi_result["overall_tbi_pct"]
                report["tbi_risk_label"]       = tbi_result["overall_risk_label"]
                report["regional_tbi_probs"]   = tbi_result["regional_probs"]
                report["tbi_figure"]           = tbi_result["figure_path"]
                report["brain3d_png"]          = tbi_result.get("brain3d_png")
                report["brain3d_mp4"]          = tbi_result.get("brain3d_mp4")
                report["brain3d_html"]         = tbi_result.get("brain3d_html")

                print(
                    f"    ↳ TBI: {tbi_result['overall_tbi_pct']:.1f}% "
                    f"[{tbi_result['overall_risk_label']}] | "
                    f"PNG: {tbi_result['figure_path']} | "
                    f"3D-render: {tbi_result.get('brain3d_png', 'N/A')} | "
                    f"3D-spin: {tbi_result.get('brain3d_mp4', 'N/A')} | "
                    f"3D-web: {tbi_result.get('brain3d_html', 'N/A')}"
                )
    else:
        print("No impacts detected — skipping Pass 2+3.\n")

    # ── RENDER annotated video ─────────────────────────────────────────────
    print("\n[Render] Writing annotated video...")

    out_path = str(Path(video_path).with_suffix("")) + "_annotated.mp4"
    writer   = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (src_w, src_h)
    )

    frame_buffer_dict = dict(tracker.frame_buffer)

    for fidx in sorted(frame_buffer_dict.keys()):
        frame       = frame_buffer_dict[fidx]
        head_states = all_frame_states.get(fidx, [])

        impact_ids  = set()
        events_now  = []
        reports_now: dict[int, dict] = {}
        for ev in all_events:
            if abs(fidx - ev.frame_idx) <= 2:
                impact_ids |= set(ev.track_ids)
                if fidx == ev.frame_idx:
                    events_now.append(ev)
                    reports_now = reports_by_event.get(ev.frame_idx, {})

        vis = _draw_frame(frame, head_states, impact_ids, fidx, events_now, reports_now)
        writer.write(vis)

        # freeze on exact impact frame for FREEZE_SECS
        if fidx in {ev.frame_idx for ev in all_events}:
            freeze_vis = vis.copy()
            h, w = freeze_vis.shape[:2]
            _label(freeze_vis, "[ IMPACT MOMENT ]", w // 2 - 110, h - 10,
                   bg=C_RED, fg=C_WHITE, scale=0.7, thick=2)
            for _ in range(freeze_frames):
                writer.write(freeze_vis)

    writer.release()
    print(f"[Render] Saved → {out_path}")

    # ── save unified JSON report ───────────────────────────────────────────
    all_reports = [r for tid_map in reports_by_event.values()
                   for r in tid_map.values()]
    json_path = str(Path(video_path).with_suffix("")) + ".impact_report.json"
    with open(json_path, "w") as f:
        json.dump({
            "events": [
                {
                    "frame":      ev.frame_idx,
                    "tracks":     ev.track_ids,
                    "confidence": ev.confidence,
                    "stages":     ev.stages,
                    "details":    ev.details,
                }
                for ev in all_events
            ],
            "profiles": all_reports,
        }, f, indent=2)

    print(f"[Done]   Report → {json_path}\n")
    return all_reports


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",    required=True)
    ap.add_argument("--window",   type=int, default=HALF_WINDOW)
    ap.add_argument("--show",     action="store_true")
    ap.add_argument("--no-depth", action="store_true",
                    help="Disable depth gate (for debugging / no GPU)")
    args = ap.parse_args()
    run(
        video_path  = args.video,
        half_window = args.window,
        show        = args.show,
        use_depth   = not args.no_depth,
    )
