"""
brain3d_vedo.py

High-quality offline 3D brain render using Vedo (VTK backend).
Produces:
  - brain3d_render.png   : static high-res render from best viewing angle
  - brain3d_spin.mp4     : 360° rotation video (~3s at 30fps)

Usage (standalone test):
    python brain3d_vedo.py --probs '{"corpus_callosum":0.78,"brainstem":0.61}'

Usage (from tbi_visualizer.py):
    from brain3d_vedo import render_brain_vedo
    render_brain_vedo(meshes, output_prefix, spin=True)
"""

from __future__ import annotations
import numpy as np
from pathlib import Path


def render_brain_vedo(
    meshes:         list[dict],
    output_prefix:  str,
    track_id:       int   = 0,
    event_frame:    int   = 0,
    overall_pct:    float = 0.0,
    spin:           bool  = True,
    spin_fps:       int   = 30,
    spin_duration:  float = 3.0,
    resolution:     tuple[int, int] = (1920, 1080),
) -> dict[str, str]:
    """
    Render brain meshes with Vedo offscreen.

    Returns dict with keys 'png' and optionally 'mp4'.
    """
    import vedo

    vedo.settings.default_backend = "vtk"
    plt = vedo.Plotter(
        offscreen = True,
        size      = resolution,
        bg        = "#0d0d1a",
        bg2       = "#1a1a2e",
    )

    vedo_meshes = []
    for m in meshes:
        verts = m["vertices"]
        faces = m["faces"]
        r, g, b = m["color_rgb"]

        mesh = vedo.Mesh([verts, faces])
        mesh.color([r, g, b])
        mesh.alpha(m["opacity"])
        mesh.lighting("plastic")
        mesh.smooth(niter=5)
        vedo_meshes.append(mesh)

    if not vedo_meshes:
        print("[Vedo] No meshes to render")
        return {}

    plt.add(vedo_meshes)

    plt.camera.SetPosition(0, -300, 100)
    plt.camera.SetFocalPoint(0, 0, 0)
    plt.camera.SetViewUp(0, 0, 1)

    title = vedo.Text2D(
        f"TBI Risk Map  |  Track #{track_id}  |  "
        f"Frame {event_frame}  |  Overall: {overall_pct:.1f}%",
        pos   = "top-center",
        s     = 0.9,
        c     = "white",
        font  = "Calco",
        bg    = "#0d0d1a",
        alpha = 0.7,
    )
    plt.add(title)

    # colorbar via a thin gradient box
    try:
        cbar = vedo.colorbar_widget(
            vedo_meshes[0], title="TBI Probability", c="white"
        )
        plt.add(cbar)
    except Exception:
        pass   # colorbar is cosmetic — skip if API unavailable

    output_prefix = str(output_prefix)
    results       = {}

    png_path = output_prefix + "_brain3d_render.png"
    plt.screenshot(png_path, scale=2)
    print(f"[Vedo] PNG saved → {png_path}")
    results["png"] = png_path

    if spin:
        mp4_path        = output_prefix + "_brain3d_spin.mp4"
        n_frames        = int(spin_fps * spin_duration)
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
                writer.write(cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR))
            writer.release()
            print(f"[Vedo] MP4 saved → {mp4_path}")
            results["mp4"] = mp4_path
        except ImportError:
            print("[Vedo] opencv not installed — skipping MP4")
        except Exception as e:
            print(f"[Vedo] MP4 failed: {e}")

    plt.close()
    return results


if __name__ == "__main__":
    import argparse, json
    from atlas_mesh_builder import AtlasMeshBuilder

    ap = argparse.ArgumentParser()
    ap.add_argument("--probs",   type=str, required=True)
    ap.add_argument("--out",     type=str, default="./test_brain")
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
