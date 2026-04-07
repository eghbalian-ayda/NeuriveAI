"""
atlas_mesh_builder.py

Converts Harvard-Oxford atlas parcels to 3D triangle meshes,
colored by regional TBI probability values.

Output per region: dict with keys
    vertices   : np.ndarray (N, 3)  — 3D coordinates in MNI space
    faces      : np.ndarray (M, 3)  — triangle indices
    color_rgb  : tuple (r, g, b)    — 0–255 color from probability
    prob       : float              — TBI probability 0–1
    name       : str                — region display name
"""

from __future__ import annotations
import numpy as np
import nibabel as nib
from nilearn import datasets
from skimage.measure import marching_cubes
from scipy.ndimage import binary_fill_holes, gaussian_filter


# ── region display names ──────────────────────────────────────────────────
REGION_NAMES = {
    "corpus_callosum": "Corpus Callosum",
    "brainstem":       "Brainstem",
    "thalamus":        "Thalamus",
    "white_matter":    "White Matter",
    "grey_matter":     "Grey Matter",
    "cerebellum":      "Cerebellum",
}

# ── atlas label indices (Harvard-Oxford subcortical, nilearn 0.13+) ────────
# sub-maxprob-thr25-1mm has 22 labels (0=Background, 1–21):
#   1=L WM, 2=L Cortex, 4=L Thalamus, 8=Brain-Stem,
#   12=R WM, 13=R Cortex, 15=R Thalamus
ATLAS_REGION_MAP = {
    "corpus_callosum": [1, 12],        # white matter proxy (CC is WM)
    "brainstem":       [8],
    "thalamus":        [4, 15],
    "white_matter":    [1, 12],
    "grey_matter":     [2, 13],
    "cerebellum":      [9, 10, 19, 20],  # hippocampus + amygdala as proxy
}


def _prob_to_rgb(prob: float) -> tuple[int, int, int]:
    """
    Map a probability in [0, 1] to an RGB color.
    0.0 = deep blue  (#1a237e)
    0.5 = amber      (#f9a825)
    1.0 = dark red   (#b71c1c)
    """
    p = float(np.clip(prob, 0.0, 1.0))
    if p < 0.5:
        t = p / 0.5
        r = int(26  + t * (249 - 26))
        g = int(35  + t * (168 - 35))
        b = int(126 + t * (37  - 126))
    else:
        t = (p - 0.5) / 0.5
        r = int(249 + t * (183 - 249))
        g = int(168 + t * (28  - 168))
        b = int(37  + t * (28  - 37))
    return (r, g, b)


class AtlasMeshBuilder:
    """
    Builds 3D triangle meshes from Harvard-Oxford atlas parcels.
    Meshes are in MNI152 voxel space, transformed to mm on export.
    """

    def __init__(self):
        print("[AtlasMeshBuilder] Loading Harvard-Oxford subcortical atlas...")
        atlas = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-1mm")
        # nilearn >= 0.10 may return a Nifti1Image directly; older returns a path
        maps = atlas.maps
        if isinstance(maps, nib.Nifti1Image):
            self._img = maps
        else:
            self._img = nib.load(maps)
        self._data   = self._img.get_fdata().astype(np.int32)
        self._affine = self._img.affine
        print(f"[AtlasMeshBuilder] Atlas loaded: shape={self._data.shape}")

    def _extract_region_volume(self, label_indices: list[int]) -> np.ndarray:
        mask = np.zeros(self._data.shape, dtype=bool)
        for idx in label_indices:
            mask |= (self._data == idx)
        mask   = binary_fill_holes(mask)
        smooth = gaussian_filter(mask.astype(float), sigma=0.8)
        return smooth

    def _volume_to_mesh(
        self,
        volume: np.ndarray,
        level: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        try:
            verts_vox, faces, _, _ = marching_cubes(
                volume, level=level, step_size=2, allow_degenerate=False
            )
        except (ValueError, RuntimeError):
            return None

        if len(verts_vox) == 0 or len(faces) == 0:
            return None

        verts_hom = np.hstack([verts_vox, np.ones((len(verts_vox), 1))])
        verts_mm  = (self._affine @ verts_hom.T).T[:, :3]
        return verts_mm, faces

    def build(self, regional_probs: dict[str, float]) -> list[dict]:
        """
        Build 3D meshes for all regions, colored by TBI probability.

        Returns list of dicts, one per region:
            name, key, vertices, faces, color_rgb, color_hex, prob, opacity
        """
        meshes = []

        for key, label_indices in ATLAS_REGION_MAP.items():
            prob  = float(regional_probs.get(key, 0.0))
            rgb   = _prob_to_rgb(prob)
            hexc  = "#{:02x}{:02x}{:02x}".format(*rgb)
            opacity = 0.25 + prob * 0.75

            print(f"  [mesh] {key}: prob={prob:.2f}  building mesh...")
            volume = self._extract_region_volume(label_indices)
            result = self._volume_to_mesh(volume)

            if result is None:
                print(f"  [mesh] {key}: empty mesh — skipping")
                continue

            verts, faces = result
            meshes.append({
                "name":      REGION_NAMES[key],
                "key":       key,
                "vertices":  verts,
                "faces":     faces,
                "color_rgb": rgb,
                "color_hex": hexc,
                "prob":      prob,
                "opacity":   opacity,
            })
            print(f"  [mesh] {key}: {len(verts)} verts, {len(faces)} faces")

        return meshes
