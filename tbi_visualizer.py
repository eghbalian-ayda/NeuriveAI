"""
tbi_visualizer.py

Takes regional MPS estimates and produces:
  1. overall_tbi_prob  : float  (0–100 %)
  2. regional_probs    : dict   {region: probability 0–1}
  3. figure saved to   : impact_<frame>_track_<id>_tbi.png

Figure layout:
  Left  — glass brain heatmap (MNI152, axial + coronal + sagittal)
  Right — horizontal bar chart of regional TBI probabilities
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for all environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from atlas_mesh_builder import AtlasMeshBuilder
from brain3d_vedo       import render_brain_vedo
from brain3d_web        import export_brain_web
from pathlib import Path
from scipy.special import expit   # sigmoid function


# ── Logistic risk curve parameters per region ──────────────────────────────
# P(injury) = sigmoid(β0 + β1 × MPS)
# Parameters calibrated so that P=0.5 at the regional MPS thresholds
# from Kleiven 2007 and Giordano & Kleiven 2014.
#
# Threshold (50% risk) MPS values from literature:
#   whole brain       : 0.26  (Kleiven 2007)
#   corpus callosum   : 0.20  (Giordano & Kleiven 2014)
#   brainstem         : 0.24  (Patton et al. 2015)
#   thalamus          : 0.22  (Giordano & Kleiven 2014)
#   white matter      : 0.27  (Kleiven 2007)
#   grey matter       : 0.32  (Kleiven 2007 — less sensitive)
#   cerebellum        : 0.30  (conservative estimate)

def _logistic_params(threshold_50: float, steepness: float = 25.0):
    """β0, β1 such that sigmoid(β0 + β1 × threshold) = 0.5"""
    b1 = steepness
    b0 = -b1 * threshold_50
    return b0, b1

RISK_CURVES = {
    "whole_brain_95pct": _logistic_params(0.26),
    "corpus_callosum":   _logistic_params(0.20),
    "brainstem":         _logistic_params(0.24),
    "thalamus":          _logistic_params(0.22),
    "white_matter":      _logistic_params(0.27),
    "grey_matter":       _logistic_params(0.32),
    "cerebellum":        _logistic_params(0.30),
}

# ── Nilearn atlas region names → our region keys ──────────────────────────
# Harvard-Oxford cortical + subcortical atlas (bundled with nilearn)
# Maps our simplified region names to atlas label indices.
# fmt: off
ATLAS_REGION_MAP = {
    "corpus_callosum": [47, 48],         # corpus callosum body + genu
    "brainstem":       [35],             # brain stem
    "thalamus":        [27, 28],         # left + right thalamus
    "white_matter":    [49, 50, 51, 52], # various WM labels
    "grey_matter":     list(range(0, 25)),
    "cerebellum":      [36, 37],
}
# fmt: on

# ── Color scheme ──────────────────────────────────────────────────────────
# Blue (low) → yellow (medium) → red (high)
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "tbi_risk",
    [(0.0, "#1a237e"),    # deep blue   — 0%
     (0.3, "#00897b"),    # teal        — 30%
     (0.5, "#f9a825"),    # amber       — 50%
     (0.7, "#e65100"),    # deep orange — 70%
     (1.0, "#b71c1c")],   # dark red    — 100%
)

RISK_COLORS = {
    "LOW":      "#43a047",
    "ELEVATED": "#fb8c00",
    "HIGH":     "#e53935",
}


def _risk_label(prob: float) -> str:
    if prob < 0.25:   return "LOW"
    elif prob < 0.50: return "ELEVATED"
    return "HIGH"


def _mps_to_probability(region: str, mps: float) -> float:
    """Apply logistic risk curve for a given region and MPS value."""
    if region not in RISK_CURVES:
        return 0.0
    b0, b1 = RISK_CURVES[region]
    return float(expit(b0 + b1 * mps))


# ─────────────────────────────────────────────────────────────────────────────
#  TBI Visualizer
# ─────────────────────────────────────────────────────────────────────────────

class TBIVisualizer:
    """
    Converts regional MPS estimates to TBI probabilities and visual outputs.
    """

    # Display names for bar chart
    DISPLAY_NAMES = {
        "corpus_callosum":   "Corpus Callosum",
        "brainstem":         "Brainstem",
        "thalamus":          "Thalamus",
        "white_matter":      "White Matter",
        "grey_matter":       "Grey Matter",
        "cerebellum":        "Cerebellum",
    }

    def __init__(self, output_dir: str = "."):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._nilearn_available = self._check_nilearn()

        # shared atlas mesh builder — loads atlas once, reused per event
        try:
            self._mesh_builder = AtlasMeshBuilder()
        except Exception as e:
            print(f"[TBIVisualizer] AtlasMeshBuilder failed: {e}")
            self._mesh_builder = None

    def _check_nilearn(self) -> bool:
        try:
            import nilearn
            return True
        except ImportError:
            print("[TBIVisualizer] nilearn not installed — "
                  "using schematic brain instead of MNI atlas")
            return False

    # ── probability computation ───────────────────────────────────────────────

    def compute_probabilities(
        self,
        regional_mps: dict[str, float],
    ) -> tuple[float, dict[str, float]]:
        """
        Compute overall and regional TBI probabilities.

        Overall probability = weighted combination of regional probabilities,
        using the same anatomical vulnerability weights as the fallback estimator.
        Highest-weight regions (corpus callosum, brainstem) contribute most.

        Returns
        -------
        overall_pct  : float   0–100
        regional     : dict    {region: probability 0–1}
        """
        # regional weights for overall combination
        region_weights = {
            "corpus_callosum": 0.30,
            "brainstem":       0.25,
            "thalamus":        0.20,
            "white_matter":    0.12,
            "grey_matter":     0.08,
            "cerebellum":      0.05,
        }

        regional_probs = {}
        weighted_sum   = 0.0
        weight_total   = 0.0

        for region, weight in region_weights.items():
            mps  = regional_mps.get(region, 0.0)
            prob = _mps_to_probability(region, mps)
            regional_probs[region] = prob
            weighted_sum  += prob * weight
            weight_total  += weight

        # overall = weighted combination, boosted by whole-brain 95th pct MPS
        whole_brain_prob = _mps_to_probability(
            "whole_brain_95pct",
            regional_mps.get("whole_brain_95pct", 0.0)
        )
        overall = 0.6 * (weighted_sum / max(weight_total, 1e-9)) + \
                  0.4 * whole_brain_prob
        overall_pct = round(float(overall) * 100, 1)

        return overall_pct, regional_probs

    # ── glass brain heatmap ───────────────────────────────────────────────────

    def _build_stat_map(self, regional_probs: dict[str, float]):
        """
        Build a NIfTI stat map from regional probabilities using
        the Harvard-Oxford atlas bundled with nilearn.
        Each atlas voxel gets the probability value of its region.
        Returns a Nifti1Image, or None if nilearn unavailable.
        """
        if not self._nilearn_available:
            return None

        try:
            from nilearn import datasets, image
            import nibabel as nib

            # load Harvard-Oxford subcortical atlas (bundled, no download)
            atlas = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
            atlas_img  = atlas.maps
            atlas_data = atlas_img.get_fdata()
            stat_data  = np.zeros_like(atlas_data, dtype=np.float32)

            for region, label_indices in ATLAS_REGION_MAP.items():
                prob = regional_probs.get(region, 0.0)
                for idx in label_indices:
                    stat_data[atlas_data == idx] = prob

            stat_img = nib.Nifti1Image(stat_data, atlas_img.affine)
            return stat_img

        except Exception as e:
            print(f"[TBIVisualizer] stat map build failed: {e}")
            return None

    def _draw_schematic_brain(
        self,
        ax: plt.Axes,
        regional_probs: dict[str, float],
        overall_pct: float,
    ) -> None:
        """
        Fallback brain visualization when nilearn is unavailable.
        Draws a simplified 2D brain schematic with labeled regional circles.
        """
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.4)
        ax.set_aspect("equal")
        ax.axis("off")

        # brain outline
        brain = plt.Circle((0, 0.1), 1.0, color="#e0e0e0", zorder=1)
        ax.add_patch(brain)

        # region positions (approximate anatomical layout)
        positions = {
            "corpus_callosum": (0.0,   0.35),
            "thalamus":        (0.0,   0.05),
            "brainstem":       (0.0,  -0.55),
            "white_matter":    (-0.45, 0.20),
            "grey_matter":     (0.45,  0.20),
            "cerebellum":      (0.0,  -0.85),
        }

        for region, (rx, ry) in positions.items():
            prob  = regional_probs.get(region, 0.0)
            color = HEATMAP_CMAP(prob)
            size  = 0.18 + prob * 0.12   # larger = higher risk
            circle = plt.Circle((rx, ry), size, color=color,
                                 zorder=2, alpha=0.85)
            ax.add_patch(circle)
            ax.text(rx, ry, f"{prob*100:.0f}%",
                    ha="center", va="center",
                    fontsize=7, fontweight="bold",
                    color="white" if prob > 0.3 else "black",
                    zorder=3)
            label = self.DISPLAY_NAMES[region].split()[0]
            ax.text(rx, ry - size - 0.08, label,
                    ha="center", va="top", fontsize=6, color="#333333",
                    zorder=3)

        ax.set_title(
            f"Brain Region Risk Map\nOverall TBI: {overall_pct:.1f}%",
            fontsize=9, fontweight="bold", pad=4
        )

    # ── bar chart ─────────────────────────────────────────────────────────────

    def _draw_bar_chart(
        self,
        ax: plt.Axes,
        regional_probs: dict[str, float],
        overall_pct: float,
    ) -> None:
        """Draw horizontal bar chart of regional TBI probabilities."""
        regions  = list(self.DISPLAY_NAMES.keys())
        labels   = [self.DISPLAY_NAMES[r] for r in regions]
        probs    = [regional_probs.get(r, 0.0) * 100 for r in regions]
        colors   = [RISK_COLORS[_risk_label(regional_probs.get(r, 0.0))]
                    for r in regions]

        y_pos = np.arange(len(regions))
        bars  = ax.barh(y_pos, probs, color=colors, edgecolor="white",
                        linewidth=0.5, height=0.6)

        # value labels
        for bar, prob in zip(bars, probs):
            ax.text(
                min(prob + 1.5, 102),
                bar.get_y() + bar.get_height() / 2,
                f"{prob:.1f}%",
                va="center", ha="left",
                fontsize=8, color="#222222"
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlim(0, 110)
        ax.set_xlabel("TBI Probability (%)", fontsize=9)
        ax.axvline(50, color="#888888", linestyle="--",
                   linewidth=0.8, alpha=0.6, label="50% threshold")
        ax.set_title(
            f"Regional TBI Probability\nOverall: {overall_pct:.1f}%",
            fontsize=9, fontweight="bold"
        )

        # legend
        legend_patches = [
            mpatches.Patch(color=RISK_COLORS["LOW"],      label="Low (<25%)"),
            mpatches.Patch(color=RISK_COLORS["ELEVATED"], label="Elevated (25–50%)"),
            mpatches.Patch(color=RISK_COLORS["HIGH"],     label="High (>50%)"),
        ]
        ax.legend(handles=legend_patches, fontsize=7,
                  loc="lower right", framealpha=0.7)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ── main interface ────────────────────────────────────────────────────────

    def visualize(
        self,
        regional_mps:  dict[str, float],
        track_id:      int,
        event_frame:   int,
        save_path:     str | None = None,
    ) -> dict:
        """
        Generate TBI probability outputs and save figure.

        Parameters
        ----------
        regional_mps  : dict from StrainEstimator.estimate()
        track_id      : ByteTrack ID of the impacted player
        event_frame   : frame index of the confirmed impact
        save_path     : optional override for output path

        Returns
        -------
        dict:
            overall_tbi_pct      : float   (0–100)
            regional_probs       : dict    {region: 0–1}
            overall_risk_label   : str     LOW / ELEVATED / HIGH
            figure_path          : str     path to saved PNG
        """
        # ── 1. compute probabilities ───────────────────────────────────────
        overall_pct, regional_probs = self.compute_probabilities(regional_mps)
        overall_risk = _risk_label(overall_pct / 100)

        # ── 2. build figure ────────────────────────────────────────────────
        fig = plt.figure(figsize=(14, 5), facecolor="#1a1a2e")

        # title bar
        risk_color = RISK_COLORS[overall_risk]
        fig.suptitle(
            f"TBI Analysis — Track #{track_id}  |  Frame {event_frame}  |  "
            f"Overall Risk: {overall_pct:.1f}%  [{overall_risk}]",
            fontsize=11, fontweight="bold",
            color="white", y=0.98
        )

        if self._nilearn_available:
            # ── left panel: nilearn glass brain ───────────────────────────
            stat_img = self._build_stat_map(regional_probs)

            if stat_img is not None:
                from nilearn import plotting

                ax_brain = fig.add_axes([0.02, 0.05, 0.50, 0.85])
                ax_brain.set_facecolor("#1a1a2e")

                display = plotting.plot_glass_brain(
                    stat_img,
                    display_mode="ortho",
                    colorbar=True,
                    vmin=0.0, vmax=1.0,
                    cmap=HEATMAP_CMAP,
                    axes=ax_brain,
                    title="",
                    annotate=False,
                    black_bg=True,
                )
            else:
                ax_brain = fig.add_subplot(121, facecolor="#1a1a2e")
                self._draw_schematic_brain(ax_brain, regional_probs,
                                           overall_pct)

        else:
            # ── fallback schematic brain ───────────────────────────────────
            ax_brain = fig.add_subplot(121, facecolor="#1a1a2e")
            self._draw_schematic_brain(ax_brain, regional_probs, overall_pct)

        # ── right panel: bar chart ─────────────────────────────────────────
        ax_bar = fig.add_subplot(122, facecolor="#1a1a2e")
        ax_bar.tick_params(colors="white")
        ax_bar.xaxis.label.set_color("white")
        ax_bar.yaxis.label.set_color("white")
        ax_bar.title.set_color("white")
        ax_bar.spines["bottom"].set_color("#555555")
        ax_bar.spines["left"].set_color("#555555")
        self._draw_bar_chart(ax_bar, regional_probs, overall_pct)

        # ── colorbar annotation ────────────────────────────────────────────
        fig.text(0.50, 0.02,
                 "Heatmap: probability of local tissue injury (0%–100%)",
                 ha="center", fontsize=7, color="#aaaaaa")

        # ── save ──────────────────────────────────────────────────────────
        if save_path is None:
            fname = f"impact_{event_frame:05d}_track_{track_id}_tbi.png"
            save_path = str(self.output_dir / fname)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)

        # ── 3D outputs ─────────────────────────────────────────────────────
        vedo_paths = {}
        web_path   = None

        if self._mesh_builder is not None:
            meshes = self._mesh_builder.build(regional_probs)

            if meshes:
                prefix = str(self.output_dir /
                             f"impact_{event_frame:05d}_track_{track_id}")

                try:
                    vedo_paths = render_brain_vedo(
                        meshes        = meshes,
                        output_prefix = prefix,
                        track_id      = track_id,
                        event_frame   = event_frame,
                        overall_pct   = overall_pct,
                        spin          = True,
                    )
                except Exception as e:
                    print(f"[TBIVisualizer] Vedo render failed: {e}")

                try:
                    web_path = export_brain_web(
                        meshes       = meshes,
                        output_path  = prefix + "_brain3d_interactive.html",
                        track_id     = track_id,
                        event_frame  = event_frame,
                        overall_pct  = overall_pct,
                    )
                except Exception as e:
                    print(f"[TBIVisualizer] Web export failed: {e}")

        return {
            "overall_tbi_pct":    overall_pct,
            "overall_risk_label": overall_risk,
            "regional_probs":     {k: round(v * 100, 1)
                                   for k, v in regional_probs.items()},
            "figure_path":        save_path,
            "brain3d_png":        vedo_paths.get("png"),
            "brain3d_mp4":        vedo_paths.get("mp4"),
            "brain3d_html":       web_path,
        }
