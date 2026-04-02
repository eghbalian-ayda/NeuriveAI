"""
plot_profiles.py — Angular velocity & acceleration profile visualiser

Reads the impact report JSON and produces a clean multi-panel figure
for each impact event, showing:
  Panel A  — ω_x, ω_y, ω_z component traces + ‖ω‖ resultant (bold)
  Panel B  — ‖α(t)‖ angular acceleration trace
  Panel C  — Brain injury metric summary (all metrics)
  Markers  — Impact moment, KLC thresholds, Δω annotation, pulse width

Usage:
    python plot_profiles.py --report test.impact_report.json
    python plot_profiles.py --report test.impact_report.json --fps 24 --save
"""

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless; switch to "TkAgg" for interactive
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from pathlib import Path


# ── colour palette ─────────────────────────────────────────────────────────────
C_X      = "#E74C3C"   # red      — ω_x
C_Y      = "#2ECC71"   # green    — ω_y
C_Z      = "#3498DB"   # blue     — ω_z
C_MAG    = "#F39C12"   # orange   — ‖ω‖
C_ALPHA  = "#9B59B6"   # purple   — ‖α‖
C_IMPACT = "#E74C3C"   # red      — impact moment line
C_BG     = "#0F1923"   # dark bg
C_PANEL  = "#192232"   # slightly lighter panel bg
C_TEXT   = "#ECF0F1"   # near-white text
C_GRID   = "#2C3E50"   # subtle grid
C_LOW    = "#2ECC71"   # green    — LOW risk
C_ELEV   = "#F39C12"   # orange   — ELEVATED risk
C_HIGH   = "#E74C3C"   # red      — HIGH risk

KLC_ELEV = 15.0        # rad/s — ELEVATED threshold
KLC_HIGH = 30.0        # rad/s — HIGH threshold


def _risk_color(label: str) -> str:
    return {"LOW": C_LOW, "ELEVATED": C_ELEV, "HIGH": C_HIGH}.get(label, C_TEXT)


def _metric_badge(ax, x, y, label, value, unit, risk, fontsize=8.5):
    """Draw a coloured metric badge at axes-fraction position (x, y)."""
    color = _risk_color(risk)
    ax.text(x, y, f"{label}", transform=ax.transAxes,
            fontsize=fontsize - 1, color=C_TEXT, ha="center", va="bottom",
            fontweight="bold")
    ax.text(x, y - 0.055, f"{value}  {unit}", transform=ax.transAxes,
            fontsize=fontsize, color=color, ha="center", va="bottom",
            fontweight="bold")
    ax.text(x, y - 0.11, risk, transform=ax.transAxes,
            fontsize=fontsize - 1.5, color=color, ha="center", va="bottom",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor=color + "33",
                      edgecolor=color, linewidth=1))


def _info_badge(ax, x, y, label, value, unit, fontsize=8.5):
    """Draw an informational metric (no risk colour) at axes-fraction (x, y)."""
    ax.text(x, y, f"{label}", transform=ax.transAxes,
            fontsize=fontsize - 1, color=C_TEXT, ha="center", va="bottom",
            fontweight="bold")
    ax.text(x, y - 0.055, f"{value}  {unit}", transform=ax.transAxes,
            fontsize=fontsize, color=C_MAG, ha="center", va="bottom",
            fontweight="bold")


def _divider(ax, y):
    ax.plot([0.05, 0.95], [y, y], color=C_GRID, lw=0.8,
            transform=ax.transAxes, clip_on=False)


def plot_event(event: dict, profiles: list[dict], fps: float,
               out_path: str | None = None):
    """
    Plot velocity + acceleration profiles for one impact event (all tracks).
    Layout per track:
        ┌────────────────────────┬──────────┐
        │   Panel A  ω(t)        │          │
        │   3-axis + resultant   │  Panel C │
        ├────────────────────────│  metrics │
        │   Panel B  ‖α(t)‖      │          │
        │   angular acceleration │          │
        └────────────────────────┴──────────┘
    """
    n_tracks = len(profiles)
    if n_tracks == 0:
        return

    event_frame = event["frame"]

    # ── figure layout ────────────────────────────────────────────────────────
    fig_h = 7.0 * n_tracks
    fig = plt.figure(figsize=(16, fig_h), facecolor=C_BG)

    outer = gridspec.GridSpec(n_tracks, 1, figure=fig,
                              hspace=0.55, left=0.06, right=0.97,
                              top=0.93, bottom=0.05)

    # ── figure title ─────────────────────────────────────────────────────────
    stage_str = " + ".join(s.upper() for s in event["stages"])
    conf_str  = f"{event['confidence']:.3f}"
    fig.suptitle(
        f"NeurivAI v2  —  Impact Event  |  Frame {event_frame}  |  "
        f"Conf {conf_str}  |  Stages: {stage_str}",
        fontsize=13, color=C_TEXT, fontweight="bold", y=0.975
    )

    for row_idx, profile in enumerate(profiles):
        tid         = profile["track_id"]
        fidxs       = np.array(profile["frame_indices"])
        omega       = np.array(profile["omega_xyz"])           # (T, 3)
        omega_mag   = np.linalg.norm(omega, axis=1)
        t_sec       = (fidxs - event_frame) / fps             # relative to impact

        omega_peak  = profile["omega_peak_rad_s"]
        bric_r      = profile["bric_r"]
        klc         = profile["klc_rot_rad_s"]
        damage      = profile["damage"]
        bric_risk   = profile["bric_r_risk"]
        klc_risk    = profile["klc_risk"]
        dmg_risk    = profile["damage_risk"]
        overall     = profile["risk_summary"]

        # new metrics (may be absent in old JSON)
        delta_omega   = profile.get("delta_omega_rad_s",  None)
        pulse_dur     = profile.get("pulse_duration_s",   None)
        alpha_vecs    = profile.get("alpha_vectors",      None)
        alpha_peak    = profile.get("alpha_peak_rad_s2",  None)

        # ── inner grid: [left_plots (2 rows)] | [right metrics (1 tall)] ─────
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 2,
            subplot_spec=outer[row_idx],
            height_ratios=[3, 2],
            width_ratios=[3.5, 1],
            wspace=0.07, hspace=0.35
        )

        # ═══════════════════════════════════════════════════════════════════════
        # Panel A — Angular velocity traces
        # ═══════════════════════════════════════════════════════════════════════
        ax = fig.add_subplot(inner[0, 0])
        ax.set_facecolor(C_PANEL)
        for spine in ax.spines.values():
            spine.set_edgecolor(C_GRID)

        ax.plot(t_sec, omega[:, 0], color=C_X,   lw=1.4, alpha=0.75, label="ω_x")
        ax.plot(t_sec, omega[:, 1], color=C_Y,   lw=1.4, alpha=0.75, label="ω_y")
        ax.plot(t_sec, omega[:, 2], color=C_Z,   lw=1.4, alpha=0.75, label="ω_z")
        ax.plot(t_sec, omega_mag,   color=C_MAG, lw=2.5, label="‖ω‖  resultant")

        # KLC threshold lines
        ax.axhline(KLC_ELEV,  color=C_ELEV, lw=0.9, ls="--", alpha=0.55)
        ax.axhline(KLC_HIGH,  color=C_HIGH, lw=0.9, ls="--", alpha=0.55)
        ax.axhline(-KLC_ELEV, color=C_ELEV, lw=0.9, ls="--", alpha=0.55)
        ax.axhline(-KLC_HIGH, color=C_HIGH, lw=0.9, ls="--", alpha=0.55)
        ax.text(t_sec[-1], KLC_ELEV + 0.3, "ELEV", color=C_ELEV,
                fontsize=6.5, ha="right", va="bottom")
        ax.text(t_sec[-1], KLC_HIGH + 0.3, "HIGH", color=C_HIGH,
                fontsize=6.5, ha="right", va="bottom")

        # impact moment
        ax.axvline(0.0, color=C_IMPACT, lw=1.8, ls="-", alpha=0.85)
        ax.text(0.02, 0.96, "IMPACT", transform=ax.transAxes,
                color=C_IMPACT, fontsize=7.5, va="top", fontweight="bold")

        # peak annotation
        peak_t = t_sec[int(np.argmax(omega_mag))]
        ax.annotate(
            f"  peak\n  {omega_peak:.1f} rad/s",
            xy=(peak_t, omega_peak),
            xytext=(peak_t + max((t_sec[-1] - t_sec[0]) * 0.05, 0.02),
                    omega_peak + 0.8),
            color=C_MAG, fontsize=7.5,
            arrowprops=dict(arrowstyle="->", color=C_MAG, lw=1.1),
        )

        # Δω span annotation (if available)
        if delta_omega is not None:
            peak_idx  = int(np.argmax(omega_mag))
            peak_val  = omega_mag[peak_idx]
            base_val  = peak_val - delta_omega
            ax.annotate(
                "", xy=(peak_t, peak_val),
                xytext=(peak_t, max(base_val, 0)),
                arrowprops=dict(arrowstyle="<->", color="#BDC3C7", lw=1.0),
            )
            ax.text(peak_t + (t_sec[-1] - t_sec[0]) * 0.03,
                    (peak_val + max(base_val, 0)) / 2,
                    f"Δω={delta_omega:.2f}", color="#BDC3C7",
                    fontsize=7, va="center")

        # pulse width shading (FWHM)
        if pulse_dur is not None and pulse_dur > 0:
            half_pw = pulse_dur / 2
            ax.axvspan(peak_t - half_pw, peak_t + half_pw,
                       color=C_MAG, alpha=0.07,
                       label=f"FWHM {pulse_dur:.3f} s")

        ax.set_ylabel("Angular velocity (rad/s)", color=C_TEXT, fontsize=8.5)
        ax.tick_params(colors=C_TEXT, labelsize=7.5)
        ax.grid(True, color=C_GRID, lw=0.4, alpha=0.5)
        ax.set_title(f"Track {tid}  —  Angular Velocity  ω(t)",
                     color=C_TEXT, fontsize=10, pad=6, fontweight="bold")

        # legend
        handles_a = [
            Line2D([0], [0], color=C_X,   lw=1.4, label="ω_x"),
            Line2D([0], [0], color=C_Y,   lw=1.4, label="ω_y"),
            Line2D([0], [0], color=C_Z,   lw=1.4, label="ω_z"),
            Line2D([0], [0], color=C_MAG, lw=2.5, label="‖ω‖  resultant"),
        ]
        if pulse_dur is not None and pulse_dur > 0:
            handles_a.append(
                Line2D([0], [0], color=C_MAG, lw=8, alpha=0.2,
                       label=f"FWHM {pulse_dur:.3f} s")
            )
        ax.legend(handles=handles_a, loc="upper left", fontsize=7.5,
                  facecolor=C_BG, edgecolor=C_GRID, labelcolor=C_TEXT,
                  framealpha=0.8)

        # ═══════════════════════════════════════════════════════════════════════
        # Panel B — Angular acceleration ‖α(t)‖
        # ═══════════════════════════════════════════════════════════════════════
        bx = fig.add_subplot(inner[1, 0])
        bx.set_facecolor(C_PANEL)
        for spine in bx.spines.values():
            spine.set_edgecolor(C_GRID)

        if alpha_vecs is not None:
            alpha_arr = np.array(alpha_vecs)          # (T, 3)
            alpha_mag = np.linalg.norm(alpha_arr, axis=1)

            bx.plot(t_sec, alpha_arr[:, 0], color=C_X,    lw=1.1, alpha=0.6, label="α_x")
            bx.plot(t_sec, alpha_arr[:, 1], color=C_Y,    lw=1.1, alpha=0.6, label="α_y")
            bx.plot(t_sec, alpha_arr[:, 2], color=C_Z,    lw=1.1, alpha=0.6, label="α_z")
            bx.plot(t_sec, alpha_mag,        color=C_ALPHA, lw=2.2, label="‖α‖  resultant")

            bx.axvline(0.0, color=C_IMPACT, lw=1.5, ls="-", alpha=0.7)

            if alpha_peak is not None:
                apeak_idx = int(np.argmax(alpha_mag))
                apeak_t   = t_sec[apeak_idx]
                bx.annotate(
                    f"  peak\n  {alpha_peak:.1f} rad/s²",
                    xy=(apeak_t, alpha_mag[apeak_idx]),
                    xytext=(apeak_t + max((t_sec[-1] - t_sec[0]) * 0.05, 0.02),
                            alpha_mag[apeak_idx] + alpha_mag.max() * 0.05),
                    color=C_ALPHA, fontsize=7.5,
                    arrowprops=dict(arrowstyle="->", color=C_ALPHA, lw=1.1),
                )

            handles_b = [
                Line2D([0], [0], color=C_X,     lw=1.1, label="α_x"),
                Line2D([0], [0], color=C_Y,     lw=1.1, label="α_y"),
                Line2D([0], [0], color=C_Z,     lw=1.1, label="α_z"),
                Line2D([0], [0], color=C_ALPHA, lw=2.2, label="‖α‖  resultant"),
            ]
            bx.legend(handles=handles_b, loc="upper left", fontsize=7,
                      facecolor=C_BG, edgecolor=C_GRID, labelcolor=C_TEXT,
                      framealpha=0.8)
        else:
            bx.text(0.5, 0.5, "α(t) not available\n(re-run pipeline)",
                    transform=bx.transAxes, color=C_TEXT + "88",
                    ha="center", va="center", fontsize=9)

        bx.set_xlabel("Time relative to impact (s)", color=C_TEXT, fontsize=8.5)
        bx.set_ylabel("Angular accel. (rad/s²)",    color=C_TEXT, fontsize=8.5)
        bx.tick_params(colors=C_TEXT, labelsize=7.5)
        bx.grid(True, color=C_GRID, lw=0.4, alpha=0.5)
        bx.set_title("Angular Acceleration  α(t) = dω/dt",
                     color=C_TEXT, fontsize=9.5, pad=5, fontweight="bold")

        # ═══════════════════════════════════════════════════════════════════════
        # Panel C — Metrics (spans both rows)
        # ═══════════════════════════════════════════════════════════════════════
        mx = fig.add_subplot(inner[:, 1])
        mx.set_facecolor(C_PANEL)
        mx.set_xticks([])
        mx.set_yticks([])
        for spine in mx.spines.values():
            spine.set_edgecolor(C_GRID)

        # overall risk header
        oc = _risk_color(overall)
        mx.text(0.5, 0.96, "OVERALL RISK", transform=mx.transAxes,
                fontsize=9, color=C_TEXT, ha="center", va="top",
                fontweight="bold")
        mx.text(0.5, 0.88, overall, transform=mx.transAxes,
                fontsize=17, color=oc, ha="center", va="top",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.35", facecolor=oc + "22",
                          edgecolor=oc, linewidth=1.5))

        _divider(mx, 0.80)

        # ── injury criteria ────────────────────────────────────────────────
        mx.text(0.5, 0.795, "INJURY METRICS", transform=mx.transAxes,
                fontsize=7.5, color=C_TEXT + "99", ha="center", va="top",
                fontweight="bold")

        _metric_badge(mx, 0.5, 0.74, "BrIC_R",
                      f"{bric_r:.3f}", "(≥0.50=HIGH)", bric_risk)
        _metric_badge(mx, 0.5, 0.59, "KLC",
                      f"{klc:.1f}", "rad/s", klc_risk)
        _metric_badge(mx, 0.5, 0.44, "DAMAGE",
                      f"{damage:.4f}", "(≥0.20=HIGH)", dmg_risk)

        _divider(mx, 0.38)

        # ── kinematic summary ──────────────────────────────────────────────
        mx.text(0.5, 0.375, "KINEMATICS", transform=mx.transAxes,
                fontsize=7.5, color=C_TEXT + "99", ha="center", va="top",
                fontweight="bold")

        _info_badge(mx, 0.5, 0.32,
                    "ω peak",
                    f"{omega_peak:.2f}", "rad/s")

        if delta_omega is not None:
            _info_badge(mx, 0.5, 0.22,
                        "Δω",
                        f"{delta_omega:.3f}", "rad/s")
        else:
            _info_badge(mx, 0.5, 0.22, "Δω", "—", "")

        if pulse_dur is not None:
            _info_badge(mx, 0.5, 0.12,
                        "Pulse FWHM",
                        f"{pulse_dur:.3f}", "s")
        else:
            _info_badge(mx, 0.5, 0.12, "Pulse FWHM", "—", "")

        if alpha_peak is not None:
            _info_badge(mx, 0.5, 0.02,
                        "α peak",
                        f"{alpha_peak:.1f}", "rad/s²")
        else:
            _info_badge(mx, 0.5, 0.02, "α peak", "—", "")

        # frames info
        mx.text(0.5, -0.04,
                f"{profile['n_frames']} frames  ·  ±{profile['n_frames']//2}f window",
                transform=mx.transAxes, fontsize=7, color=C_TEXT + "77",
                ha="center", va="bottom")

    # ── save / show ──────────────────────────────────────────────────────────
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=C_BG)
        print(f"  Saved: {out_path}")
    else:
        plt.show()

    plt.close(fig)


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True,
                    help="Path to .impact_report.json")
    ap.add_argument("--fps", type=float, default=None,
                    help="Video FPS (read from report if omitted)")
    ap.add_argument("--save", action="store_true",
                    help="Save PNGs alongside the report (default: show)")
    args = ap.parse_args()

    with open(args.report) as f:
        data = json.load(f)

    events   = data["events"]
    profiles = data["profiles"]
    fps      = args.fps or 30.0

    print(f"\nNeurivAI — Velocity Profile Plotter")
    print(f"Report : {args.report}")
    print(f"Events : {len(events)}  |  Profiles : {len(profiles)}\n")

    for ev in events:
        ev_frame = ev["frame"]
        ev_profiles = [p for p in profiles if p["event_frame"] == ev_frame]

        if not ev_profiles:
            print(f"  [!] No profiles for event @ frame {ev_frame}")
            continue

        print(f"  Plotting event @ frame {ev_frame}  "
              f"(tracks {[p['track_id'] for p in ev_profiles]})...")

        out_path = None
        if args.save:
            base     = Path(args.report).with_suffix("")
            out_path = str(base) + f"_profiles_frame{ev_frame:05d}.png"

        plot_event(ev, ev_profiles, fps, out_path)

    if not args.save:
        print("\nPlots displayed. Pass --save to write PNG files.")
    else:
        print("\nDone.")


if __name__ == "__main__":
    main()
