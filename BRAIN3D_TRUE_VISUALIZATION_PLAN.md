# True 3D Brain Visualization — Claude Code Implementation Plan
# Two outputs from the same pipeline data:
#   1. Vedo — high-quality offline 3D render → PNG/MP4 screenshot
#   2. Three.js Mesh3d — interactive web demo → standalone HTML

---

## Overview

Both outputs consume the exact same input:
    regional_probs dict from TBIVisualizer.compute_probabilities()
    e.g. {"corpus_callosum": 0.78, "brainstem": 0.61, ...}

The shared pipeline is:
    regional_probs
        │
        ▼
    AtlasMeshBuilder          ← NEW shared utility
    Extracts each brain region from Harvard-Oxford NIfTI atlas,
    runs marching cubes to produce a 3D triangle mesh per region,
    colors it by TBI probability.
        │
        ├──► Vedo renderer    → brain3d_render.png  +  brain3d_spin.mp4
        └──► Three.js exporter→ brain3d_interactive.html

---

## Dependencies

```bash
pip install vedo          # 3D mesh rendering, VTK-based
pip install pyvista       # VTK wrapper (used by AtlasMeshBuilder for meshing)
pip install scikit-image  # marching_cubes algorithm
pip install nibabel       # NIfTI atlas loading (already installed via nilearn)
pip install nilearn       # atlas download (already installed)
pip install trimesh       # mesh export to OBJ/JSON for Three.js
pip install numpy scipy   # already installed
```

---

## File 1 — `atlas_mesh_builder.py`  ← NEW shared utility

**Purpose:** Loads the Harvard-Oxford subcortical atlas, extracts each
brain region as a binary volume, runs marching cubes to produce a
watertight 3D triangle mesh, and returns colored mesh objects.
Used by both the Vedo renderer and the Three.js exporter.

```python
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

# ── atlas label indices (Harvard-Oxford subcortical) ──────────────────────
ATLAS_REGION_MAP = {
    "corpus_callosum": [47, 48],
    "brainstem":       [35],
    "thalamus":        [27, 28],
    "white_matter":    [49, 50, 51, 52],
    "grey_matter":     list(range(1, 25)),
    "cerebellum":      [36, 37],
}

# ── TBI probability → RGB color (blue→yellow→red) ─────────────────────────
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
        # load atlas once — cached by nilearn after first download
        print("[AtlasMeshBuilder] Loading Harvard-Oxford subcortical atlas...")
        atlas      = datasets.fetch_atlas_harvard_oxford(
            "sub-maxprob-thr25-1mm"   # 1mm for better mesh quality
        )
        self._img  = nib.load(atlas.maps)
        self._data = self._img.get_fdata().astype(np.int32)
        self._affine = self._img.affine
        print(f"[AtlasMeshBuilder] Atlas loaded: shape={self._data.shape}")

    def _extract_region_volume(self, label_indices: list[int]) -> np.ndarray:
        """
        Return a binary 3D volume mask for the union of given atlas labels.
        Applies hole-filling and light Gaussian smoothing for cleaner meshes.
        """
        mask = np.zeros(self._data.shape, dtype=bool)
        for idx in label_indices:
            mask |= (self._data == idx)
        # fill internal holes for watertight mesh
        mask = binary_fill_holes(mask)
        # smooth boundary slightly to reduce staircase artifacts
        smooth = gaussian_filter(mask.astype(float), sigma=0.8)
        return smooth

    def _volume_to_mesh(
        self,
        volume: np.ndarray,
        level: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Run marching cubes on a binary volume.
        Transforms vertices from voxel coords to MNI mm coords via affine.

        Returns (vertices, faces) or None if mesh is empty.
        """
        try:
            verts_vox, faces, _, _ = marching_cubes(
                volume, level=level, step_size=2, allow_degenerate=False
            )
        except (ValueError, RuntimeError):
            return None

        if len(verts_vox) == 0 or len(faces) == 0:
            return None

        # transform vertices from voxel space to MNI mm space
        verts_hom = np.hstack([verts_vox,
                                np.ones((len(verts_vox), 1))])
        verts_mm  = (self._affine @ verts_hom.T).T[:, :3]

        return verts_mm, faces

    def build(
        self,
        regional_probs: dict[str, float],
    ) -> list[dict]:
        """
        Build 3D meshes for all regions, colored by TBI probability.

        Parameters
        ----------
        regional_probs : dict {region_key: probability 0–1}
                         from TBIVisualizer.compute_probabilities()

        Returns
        -------
        list of dicts, one per region:
            name       : str   display name
            key        : str   internal key
            vertices   : ndarray (N, 3)  MNI mm coords
            faces      : ndarray (M, 3)  triangle indices
            color_rgb  : tuple (r, g, b) 0–255
            color_hex  : str   '#rrggbb'
            prob       : float TBI probability
            opacity    : float 0–1  (lower prob → more transparent)
        """
        meshes = []

        for key, label_indices in ATLAS_REGION_MAP.items():
            prob  = float(regional_probs.get(key, 0.0))
            rgb   = _prob_to_rgb(prob)
            hexc  = "#{:02x}{:02x}{:02x}".format(*rgb)

            # opacity: high-risk regions are solid, low-risk are translucent
            # this helps see the internal structures when rendered together
            opacity = 0.25 + prob * 0.75   # range 0.25–1.0

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
```

---

## File 2 — `brain3d_vedo.py`  ← NEW offline renderer

**Purpose:** Takes the mesh list from AtlasMeshBuilder and renders it
as a high-quality 3D image using Vedo (VTK-based).
Produces a PNG screenshot and a rotating MP4 video.
Runs headless — no display required.

```python
"""
brain3d_vedo.py

High-quality offline 3D brain render using Vedo (VTK backend).
Produces:
  - brain3d_render.png   : static high-res render from best viewing angle
  - brain3d_spin.mp4     : 360° rotation video (optional, ~3s at 30fps)

Usage (standalone test):
    python brain3d_vedo.py --probs '{"corpus_callosum":0.78,"brainstem":0.61}'

Usage (from tbi_visualizer.py):
    from brain3d_vedo import render_brain_vedo
    render_brain_vedo(meshes, output_prefix, spin=True)
"""

from __future__ import annotations
import os
import numpy as np
from pathlib import Path


def render_brain_vedo(
    meshes:         list[dict],
    output_prefix:  str,
    track_id:       int  = 0,
    event_frame:    int  = 0,
    overall_pct:    float = 0.0,
    spin:           bool = True,
    spin_fps:       int  = 30,
    spin_duration:  float = 3.0,
    resolution:     tuple[int, int] = (1920, 1080),
) -> dict[str, str]:
    """
    Render brain meshes with Vedo.

    Parameters
    ----------
    meshes          : list of mesh dicts from AtlasMeshBuilder.build()
    output_prefix   : path prefix for output files (no extension)
    track_id        : for title annotation
    event_frame     : for title annotation
    overall_pct     : overall TBI probability % for title
    spin            : whether to render 360° rotation video
    spin_fps        : video frame rate
    spin_duration   : video length in seconds
    resolution      : render resolution (width, height)

    Returns
    -------
    dict with keys 'png' and optionally 'mp4' — paths to saved files
    """
    import vedo

    # ── set up offscreen rendering ─────────────────────────────────────────
    vedo.settings.default_backend = "vtk"
    plt = vedo.Plotter(
        offscreen = True,
        size      = resolution,
        bg        = "#0d0d1a",    # very dark navy background
        bg2       = "#1a1a2e",    # subtle gradient
    )

    # ── build vedo mesh objects ────────────────────────────────────────────
    vedo_meshes = []
    for m in meshes:
        verts = m["vertices"]
        faces = m["faces"]
        r, g, b = m["color_rgb"]

        mesh = vedo.Mesh([verts, faces])
        mesh.color([r, g, b])
        mesh.alpha(m["opacity"])
        mesh.lighting("plastic")   # slight specular highlight
        mesh.smooth(niter=5)       # final mesh smoothing for quality
        vedo_meshes.append(mesh)

    if not vedo_meshes:
        print("[Vedo] No meshes to render")
        return {}

    plt.add(vedo_meshes)

    # ── camera position: slightly elevated front-left view ────────────────
    # MNI space: x=left-right, y=posterior-anterior, z=inferior-superior
    plt.camera.SetPosition(0, -300, 100)
    plt.camera.SetFocalPoint(0, 0, 0)
    plt.camera.SetViewUp(0, 0, 1)

    # ── title annotation ──────────────────────────────────────────────────
    title = vedo.Text2D(
        f"TBI Risk Map  |  Track #{track_id}  |  "
        f"Frame {event_frame}  |  Overall: {overall_pct:.1f}%",
        pos="top-center",
        s=0.9,
        c="white",
        font="Calco",
        bg="#0d0d1a",
        alpha=0.7,
    )
    plt.add(title)

    # ── colorbar ──────────────────────────────────────────────────────────
    # manual gradient colorbar using a dummy scalar mesh
    cbar_mesh = vedo.Mesh([np.array([[0,0,0],[1,0,0],[0,1,0]]),
                            np.array([[0,1,2]])])
    cbar_mesh.cmap("RdYlBu_r", np.linspace(0, 1, 100))
    plt.add_colorbar(
        cbar_mesh,
        title = "TBI Probability",
        pos   = (0.85, 0.1),
        c     = "white",
    )

    output_prefix = str(output_prefix)
    results       = {}

    # ── static PNG render ─────────────────────────────────────────────────
    png_path = output_prefix + "_brain3d_render.png"
    plt.screenshot(png_path, scale=2)   # 2× scale for high DPI
    print(f"[Vedo] PNG saved → {png_path}")
    results["png"] = png_path

    # ── 360° rotation video ────────────────────────────────────────────────
    if spin:
        mp4_path  = output_prefix + "_brain3d_spin.mp4"
        n_frames  = int(spin_fps * spin_duration)
        angle_per_frame = 360.0 / n_frames

        try:
            import cv2
            writer = cv2.VideoWriter(
                mp4_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                spin_fps,
                resolution,
            )

            for i in range(n_frames):
                plt.camera.Azimuth(angle_per_frame)
                frame_img = plt.screenshot(asarray=True)
                # vedo returns RGB, cv2 needs BGR
                writer.write(cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR))

            writer.release()
            print(f"[Vedo] MP4 saved → {mp4_path}")
            results["mp4"] = mp4_path

        except ImportError:
            print("[Vedo] opencv not installed — skipping MP4 generation")
        except Exception as e:
            print(f"[Vedo] MP4 generation failed: {e}")

    plt.close()
    return results


# ── standalone test entry point ───────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json
    from atlas_mesh_builder import AtlasMeshBuilder

    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", type=str, required=True,
                    help='JSON dict e.g. \'{"corpus_callosum":0.78}\'')
    ap.add_argument("--out",   type=str, default="./test_brain")
    ap.add_argument("--no-spin", action="store_true")
    args = ap.parse_args()

    probs   = json.loads(args.probs)
    builder = AtlasMeshBuilder()
    meshes  = builder.build(probs)

    render_brain_vedo(
        meshes        = meshes,
        output_prefix = args.out,
        overall_pct   = sum(probs.values()) / len(probs) * 100,
        spin          = not args.no_spin,
    )
```

---

## File 3 — `brain3d_web.py`  ← NEW interactive HTML exporter

**Purpose:** Takes the same mesh list and exports a self-contained HTML
file using Three.js with OrbitControls. The user can rotate, zoom, and
pan the brain in any browser with no server required.
Each brain region is a separate Three.js Mesh with its own color and opacity.

```python
"""
brain3d_web.py

Exports an interactive 3D brain visualization as a standalone HTML file.
Uses Three.js (loaded from CDN) with OrbitControls for rotation/zoom/pan.

Each brain region is a separate colored 3D mesh.
Opacity encodes TBI probability (high risk = solid, low risk = translucent).
A legend panel shows each region name and probability value.

No server needed — open the HTML file directly in any browser.
"""

from __future__ import annotations
import json
import numpy as np
from pathlib import Path


def _mesh_to_js_geometry(mesh: dict) -> str:
    """
    Serialize one mesh to Three.js BufferGeometry JavaScript source.
    Returns a JS string that creates a THREE.Mesh and adds it to the scene.
    """
    verts  = mesh["vertices"].flatten().tolist()
    faces  = mesh["faces"].flatten().tolist()
    r, g, b = [c / 255.0 for c in mesh["color_rgb"]]
    opacity = mesh["opacity"]
    name    = mesh["name"]
    prob    = mesh["prob"]
    hexc    = mesh["color_hex"]

    # serialize as compact JSON for embedding in JS
    verts_json = json.dumps(verts)
    faces_json = json.dumps(faces)

    return f"""
    {{
        // ── {name} (prob={prob:.2f}) ──────────────────────────────────────
        const geo_{id(mesh)} = new THREE.BufferGeometry();
        const vertices_{id(mesh)} = new Float32Array({verts_json});
        const indices_{id(mesh)}  = new Uint32Array({faces_json});
        geo_{id(mesh)}.setAttribute(
            'position',
            new THREE.BufferAttribute(vertices_{id(mesh)}, 3)
        );
        geo_{id(mesh)}.setIndex(
            new THREE.BufferAttribute(indices_{id(mesh)}, 1)
        );
        geo_{id(mesh)}.computeVertexNormals();

        const mat_{id(mesh)} = new THREE.MeshPhongMaterial({{
            color:       0x{mesh["color_hex"][1:]},
            opacity:     {opacity:.3f},
            transparent: true,
            shininess:   40,
            side:        THREE.DoubleSide,
        }});

        const mesh_{id(mesh)} = new THREE.Mesh(geo_{id(mesh)}, mat_{id(mesh)});
        scene.add(mesh_{id(mesh)});
        regionMeshes.push({{ name: "{name}", mesh: mesh_{id(mesh)}, prob: {prob:.3f}, color: "{hexc}" }});
    }}
    """


def export_brain_web(
    meshes:        list[dict],
    output_path:   str,
    track_id:      int   = 0,
    event_frame:   int   = 0,
    overall_pct:   float = 0.0,
) -> str:
    """
    Export brain meshes to a self-contained interactive HTML file.

    Parameters
    ----------
    meshes       : list of mesh dicts from AtlasMeshBuilder.build()
    output_path  : full path for the output HTML file
    track_id     : for header annotation
    event_frame  : for header annotation
    overall_pct  : overall TBI % for header

    Returns
    -------
    str : path to saved HTML file
    """
    # compute scene center from all vertices for camera targeting
    all_verts = np.vstack([m["vertices"] for m in meshes])
    center    = all_verts.mean(axis=0).tolist()
    max_range = float(np.abs(all_verts).max())

    # generate JS geometry blocks for each region
    geometry_blocks = "\n".join(_mesh_to_js_geometry(m) for m in meshes)

    # legend data for sidebar
    legend_items = json.dumps([
        {
            "name":  m["name"],
            "prob":  round(m["prob"] * 100, 1),
            "color": m["color_hex"],
            "risk":  "HIGH" if m["prob"] >= 0.5
                     else "ELEVATED" if m["prob"] >= 0.25
                     else "LOW",
        }
        for m in sorted(meshes, key=lambda x: -x["prob"])
    ])

    risk_color = (
        "#e53935" if overall_pct >= 50 else
        "#fb8c00" if overall_pct >= 25 else
        "#43a047"
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NeurivAI — TBI Brain Map | Track #{track_id} | Frame {event_frame}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    background: #0d0d1a;
    color: #e2e8f0;
    font-family: 'JetBrains Mono', 'Courier New', monospace;
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
  }}

  /* ── header ── */
  #header {{
    background: #10141c;
    border-bottom: 1px solid #1e2535;
    padding: 10px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-shrink: 0;
  }}
  #header h1 {{
    font-size: 0.85rem;
    color: #00d4ff;
    letter-spacing: 0.08em;
    font-weight: 700;
  }}
  #header .overall {{
    font-size: 1.1rem;
    font-weight: 700;
    color: {risk_color};
    letter-spacing: 0.05em;
  }}
  #header .meta {{
    font-size: 0.7rem;
    color: #64748b;
  }}

  /* ── layout ── */
  #main {{
    display: flex;
    flex: 1;
    overflow: hidden;
  }}

  /* ── 3D canvas ── */
  #canvas-container {{
    flex: 1;
    position: relative;
  }}
  #canvas-container canvas {{
    display: block;
    width: 100% !important;
    height: 100% !important;
  }}
  #controls-hint {{
    position: absolute;
    bottom: 12px;
    left: 50%;
    transform: translateX(-50%);
    font-size: 0.65rem;
    color: #475569;
    letter-spacing: 0.05em;
    pointer-events: none;
  }}

  /* ── sidebar ── */
  #sidebar {{
    width: 240px;
    background: #10141c;
    border-left: 1px solid #1e2535;
    overflow-y: auto;
    padding: 16px 12px;
    flex-shrink: 0;
  }}
  #sidebar h2 {{
    font-size: 0.72rem;
    color: #64748b;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 14px;
  }}
  .region-item {{
    margin-bottom: 10px;
    padding: 8px 10px;
    background: #161b27;
    border-radius: 8px;
    border-left: 3px solid transparent;
    cursor: pointer;
    transition: background 0.15s;
  }}
  .region-item:hover {{ background: #1e2535; }}
  .region-name {{
    font-size: 0.72rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 4px;
  }}
  .region-bar-wrap {{
    background: #0d0d1a;
    border-radius: 4px;
    height: 6px;
    overflow: hidden;
    margin-bottom: 4px;
  }}
  .region-bar {{
    height: 100%;
    border-radius: 4px;
    transition: width 0.3s;
  }}
  .region-prob {{
    font-size: 0.65rem;
    color: #94a3b8;
    display: flex;
    justify-content: space-between;
  }}
  .risk-HIGH     {{ color: #e53935; }}
  .risk-ELEVATED {{ color: #fb8c00; }}
  .risk-LOW      {{ color: #43a047; }}

  /* ── colorbar ── */
  #colorbar {{
    margin-top: 20px;
    padding-top: 14px;
    border-top: 1px solid #1e2535;
  }}
  #colorbar h2 {{
    font-size: 0.65rem;
    color: #64748b;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 8px;
  }}
  #colorbar-gradient {{
    height: 12px;
    border-radius: 6px;
    background: linear-gradient(to right, #1a237e, #00897b, #f9a825, #e65100, #b71c1c);
    margin-bottom: 4px;
  }}
  .cbar-labels {{
    display: flex;
    justify-content: space-between;
    font-size: 0.6rem;
    color: #64748b;
  }}
</style>
</head>
<body>

<div id="header">
  <div>
    <h1>NeurivAI — TBI Brain Risk Map</h1>
    <div class="meta">Track #{track_id} &nbsp;|&nbsp; Frame {event_frame:05d}</div>
  </div>
  <div class="overall">{overall_pct:.1f}% Overall TBI Risk</div>
</div>

<div id="main">
  <div id="canvas-container">
    <div id="controls-hint">🖱 Drag to rotate &nbsp;·&nbsp; Scroll to zoom &nbsp;·&nbsp; Right-drag to pan</div>
  </div>

  <div id="sidebar">
    <h2>Regional Risk</h2>
    <div id="legend"></div>
    <div id="colorbar">
      <h2>Probability Scale</h2>
      <div id="colorbar-gradient"></div>
      <div class="cbar-labels"><span>0%</span><span>50%</span><span>100%</span></div>
    </div>
  </div>
</div>

<!-- Three.js from CDN -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
// ── OrbitControls (inline, r128 compatible) ──────────────────────────────
// Minimal OrbitControls implementation — avoids needing a separate file
(function() {{
  THREE.OrbitControls = function(camera, domElement) {{
    this.camera = camera;
    this.domElement = domElement;
    this.enableDamping = true;
    this.dampingFactor = 0.05;
    this.rotateSpeed = 0.8;
    this.zoomSpeed = 1.2;

    let isMouseDown = false, isRightDown = false;
    let lastX = 0, lastY = 0;
    const spherical = new THREE.Spherical().setFromVector3(
      camera.position.clone().sub(new THREE.Vector3(...{center}))
    );
    const target = new THREE.Vector3(...{center});

    domElement.addEventListener('mousedown', e => {{
      isMouseDown = e.button === 0;
      isRightDown = e.button === 2;
      lastX = e.clientX; lastY = e.clientY;
    }});
    domElement.addEventListener('mouseup',   () => {{ isMouseDown = false; isRightDown = false; }});
    domElement.addEventListener('contextmenu', e => e.preventDefault());
    domElement.addEventListener('mousemove', e => {{
      const dx = e.clientX - lastX;
      const dy = e.clientY - lastY;
      lastX = e.clientX; lastY = e.clientY;
      if (isMouseDown) {{
        spherical.theta -= dx * 0.005 * this.rotateSpeed;
        spherical.phi   = Math.max(0.1, Math.min(Math.PI - 0.1,
                          spherical.phi + dy * 0.005 * this.rotateSpeed));
      }}
      if (isRightDown) {{
        const pan = new THREE.Vector3(-dx * 0.3, dy * 0.3, 0);
        pan.applyQuaternion(camera.quaternion);
        target.add(pan);
      }}
    }});
    domElement.addEventListener('wheel', e => {{
      spherical.radius = Math.max(50, Math.min(800,
        spherical.radius + e.deltaY * 0.3 * this.zoomSpeed));
    }});

    this.update = function() {{
      camera.position.setFromSpherical(spherical).add(target);
      camera.lookAt(target);
    }};
  }};
}})();

// ── Scene setup ───────────────────────────────────────────────────────────
const container = document.getElementById('canvas-container');
const scene     = new THREE.Scene();
scene.background = new THREE.Color(0x0d0d1a);
scene.fog        = new THREE.FogExp2(0x0d0d1a, 0.0008);

const camera = new THREE.PerspectiveCamera(
  50,
  container.clientWidth / container.clientHeight,
  0.1,
  2000
);
camera.position.set(0, {-max_range * 2.2:.1f}, {max_range * 0.5:.1f});
camera.lookAt(...{center});

const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: false }});
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.shadowMap.enabled = true;
container.appendChild(renderer.domElement);

// ── Lighting ──────────────────────────────────────────────────────────────
const ambient = new THREE.AmbientLight(0xffffff, 0.4);
scene.add(ambient);
const key   = new THREE.DirectionalLight(0xffffff, 0.8);
key.position.set(100, -200, 200);
scene.add(key);
const fill  = new THREE.DirectionalLight(0x8899ff, 0.3);
fill.position.set(-100, 100, -100);
scene.add(fill);
const back  = new THREE.DirectionalLight(0xffeedd, 0.2);
back.position.set(0, 200, -200);
scene.add(back);

// ── Meshes ────────────────────────────────────────────────────────────────
const regionMeshes = [];
{geometry_blocks}

// ── Controls ──────────────────────────────────────────────────────────────
const controls = new THREE.OrbitControls(camera, renderer.domElement);

// ── Animation loop ────────────────────────────────────────────────────────
function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}}
animate();

// ── Resize handler ────────────────────────────────────────────────────────
window.addEventListener('resize', () => {{
  camera.aspect = container.clientWidth / container.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(container.clientWidth, container.clientHeight);
}});

// ── Build sidebar legend ──────────────────────────────────────────────────
const legendData = {legend_items};
const legendEl   = document.getElementById('legend');

legendData.forEach(item => {{
  const div = document.createElement('div');
  div.className = 'region-item';
  div.style.borderLeftColor = item.color;
  div.innerHTML = `
    <div class="region-name">${{item.name}}</div>
    <div class="region-bar-wrap">
      <div class="region-bar"
           style="width:${{item.prob}}%;background:${{item.color}}"></div>
    </div>
    <div class="region-prob">
      <span>${{item.prob.toFixed(1)}}%</span>
      <span class="risk-${{item.risk}}">${{item.risk}}</span>
    </div>
  `;
  legendEl.appendChild(div);
}});
</script>
</body>
</html>"""

    out_path = str(output_path)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[Brain3D Web] Interactive HTML saved → {out_path}")
    return out_path
```

---

## File 4 — Modifications to `tbi_visualizer.py`

Replace the `_build_3d_html()` method (if it exists from previous plan)
and add calls to both new renderers inside `visualize()`.

### At the top of `tbi_visualizer.py` add imports:

```python
# add after existing imports
from atlas_mesh_builder import AtlasMeshBuilder
from brain3d_vedo       import render_brain_vedo
from brain3d_web        import export_brain_web
```

### Add class-level atlas builder (initialise once):

```python
class TBIVisualizer:

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
```

### Inside `visualize()`, after the PNG is saved, add:

```python
# ── 3D outputs ──────────────────────────────────────────────────────────
vedo_paths = {}
web_path   = None

if self._mesh_builder is not None:
    # build meshes once — shared by both renderers
    meshes = self._mesh_builder.build(regional_probs)

    if meshes:
        prefix = str(self.output_dir /
                     f"impact_{event_frame:05d}_track_{track_id}")

        # Vedo: offline high-quality PNG + MP4
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

        # Three.js: interactive web HTML
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

# add to return dict
return {{
    "overall_tbi_pct":    overall_pct,
    "overall_risk_label": overall_risk,
    "regional_probs":     {{k: round(v * 100, 1)
                           for k, v in regional_probs.items()}},
    "figure_path":        save_path,              # existing 2D summary PNG
    "brain3d_png":        vedo_paths.get("png"),  # NEW: high-quality render
    "brain3d_mp4":        vedo_paths.get("mp4"),  # NEW: 360° spin video
    "brain3d_html":       web_path,               # NEW: interactive web
}}
```

---

## Output Per Impact Event (complete)

```
competition/ (or video directory)
│
├── impact_01247_track_3_tbi.png               ← existing: 2D summary
│
├── impact_01247_track_3_brain3d_render.png    ← NEW: Vedo high-res render
├── impact_01247_track_3_brain3d_spin.mp4      ← NEW: Vedo 360° rotation
└── impact_01247_track_3_brain3d_interactive.html  ← NEW: Three.js web demo
```

### What each output looks like

**`_brain3d_render.png`**
Static 1920×1080 render. Dark navy background. Brain regions as 3D
colored meshes — corpus callosum in red/orange if high risk, brainstem
behind it, thalamus visible inside. Specular highlights. Slightly
elevated front-left camera angle. Title bar with track ID and overall %.

**`_brain3d_spin.mp4`**
3-second 360° rotation of the same scene at 30fps. Brain rotates around
the vertical axis. Good for presentations and demos.

**`_brain3d_interactive.html`**
Open in any browser. Dark theme. Left panel: rotatable 3D brain with
OrbitControls (drag to rotate, scroll to zoom, right-drag to pan).
Right panel: sidebar showing each region's name, probability bar, and
risk label. Header shows overall TBI % with color coded risk. Works
completely offline after the Three.js CDN loads once.

---

## Final File Structure

```
competition/
├── tbi_visualizer.py          ← MODIFIED (add mesh builder calls)
├── atlas_mesh_builder.py      ← NEW
├── brain3d_vedo.py            ← NEW
├── brain3d_web.py             ← NEW
│
│   ── all other files unchanged ──
├── track_video.py
├── strain_estimator.py
├── head_tracker.py
... etc
```

---

## Testing Standalone (before integrating with full pipeline)

```bash
# test atlas mesh builder
python atlas_mesh_builder.py   # should print mesh sizes per region

# test Vedo render with dummy probabilities
python brain3d_vedo.py \
  --probs '{"corpus_callosum":0.78,"brainstem":0.61,"thalamus":0.55,"white_matter":0.48,"grey_matter":0.31,"cerebellum":0.22}' \
  --out ./test_brain

# produces: test_brain_brain3d_render.png + test_brain_brain3d_spin.mp4

# test web export (import and call directly — no CLI for web module)
python -c "
from atlas_mesh_builder import AtlasMeshBuilder
from brain3d_web import export_brain_web
probs = {'corpus_callosum':0.78,'brainstem':0.61,'thalamus':0.55,
         'white_matter':0.48,'grey_matter':0.31,'cerebellum':0.22}
b = AtlasMeshBuilder()
meshes = b.build(probs)
export_brain_web(meshes, './test_brain3d_interactive.html',
                 track_id=3, event_frame=1247, overall_pct=67.4)
"
# then open test_brain3d_interactive.html in a browser
```
