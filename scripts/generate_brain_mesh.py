"""
scripts/generate_brain_mesh.py

One-time setup script: generates the static brain region geometry file used
by the 3D brain visualizer in the web frontend.

Requires: nibabel, nilearn, scikit-image, scipy
  pip install nibabel nilearn scikit-image

Output: web/frontend/public/brain_regions.json
  A JSON array of 6 region objects, each with:
    key       : str               — region identifier
    name      : str               — display name
    vertices  : [[x,y,z], ...]   — Nx3 MNI mm coordinates (float32, rounded)
    faces     : [[i,j,k], ...]   — Mx3 triangle indices (int)

Run from the repo root:
  python scripts/generate_brain_mesh.py
"""

from __future__ import annotations
import sys
import json
import numpy as np
from pathlib import Path

# resolve repo root regardless of cwd
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from atlas_mesh_builder import AtlasMeshBuilder, REGION_NAMES

OUTPUT = REPO_ROOT / "web" / "frontend" / "public" / "brain_regions.json"


def main() -> None:
    print("[generate_brain_mesh] Building Harvard-Oxford atlas meshes...")

    # Pass uniform prob=0.5 so we get full meshes for every region.
    # The frontend applies colors dynamically at render time.
    uniform_probs = {key: 0.5 for key in REGION_NAMES}

    builder = AtlasMeshBuilder()
    meshes  = builder.build(uniform_probs)

    output_data = []
    for m in meshes:
        # Round vertices to 2 decimal places to keep JSON compact (~5–15 MB → ~2–5 MB)
        verts_rounded = np.round(m["vertices"].astype(np.float32), 2).tolist()
        faces_list    = m["faces"].astype(int).tolist()
        output_data.append({
            "key":      m["key"],
            "name":     m["name"],
            "vertices": verts_rounded,
            "faces":    faces_list,
        })
        print(f"  {m['key']:20s}  {len(verts_rounded):6d} verts  {len(faces_list):6d} faces")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(output_data, f, separators=(",", ":"))   # compact (no spaces)

    size_mb = OUTPUT.stat().st_size / 1_048_576
    print(f"\n[generate_brain_mesh] Saved → {OUTPUT}  ({size_mb:.1f} MB)")
    print("[generate_brain_mesh] Done. Commit web/frontend/public/brain_regions.json.")


if __name__ == "__main__":
    main()
