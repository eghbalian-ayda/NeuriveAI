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


def _mesh_to_js_geometry(mesh: dict, uid: int) -> str:
    """Serialize one mesh to Three.js BufferGeometry JavaScript source."""
    verts   = mesh["vertices"].flatten().tolist()
    faces   = mesh["faces"].flatten().tolist()
    opacity = mesh["opacity"]
    name    = mesh["name"]
    prob    = mesh["prob"]
    hexc    = mesh["color_hex"]

    verts_json = json.dumps(verts)
    faces_json = json.dumps(faces)

    return f"""
    {{
        // ── {name} (prob={prob:.2f}) ──
        const geo_{uid} = new THREE.BufferGeometry();
        const vertices_{uid} = new Float32Array({verts_json});
        const indices_{uid}  = new Uint32Array({faces_json});
        geo_{uid}.setAttribute('position', new THREE.BufferAttribute(vertices_{uid}, 3));
        geo_{uid}.setIndex(new THREE.BufferAttribute(indices_{uid}, 1));
        geo_{uid}.computeVertexNormals();

        const mat_{uid} = new THREE.MeshPhongMaterial({{
            color:       0x{hexc[1:]},
            opacity:     {opacity:.3f},
            transparent: true,
            shininess:   40,
            side:        THREE.DoubleSide,
        }});

        const mesh_{uid} = new THREE.Mesh(geo_{uid}, mat_{uid});
        scene.add(mesh_{uid});
        regionMeshes.push({{ name: "{name}", mesh: mesh_{uid}, prob: {prob:.3f}, color: "{hexc}" }});
    }}
    """


def export_brain_web(
    meshes:      list[dict],
    output_path: str,
    track_id:    int   = 0,
    event_frame: int   = 0,
    overall_pct: float = 0.0,
) -> str:
    """
    Export brain meshes to a self-contained interactive HTML file.
    Returns path to saved HTML.
    """
    all_verts = np.vstack([m["vertices"] for m in meshes])
    center    = all_verts.mean(axis=0).tolist()
    max_range = float(np.abs(all_verts).max())

    geometry_blocks = "\n".join(
        _mesh_to_js_geometry(m, i) for i, m in enumerate(meshes)
    )

    legend_items = json.dumps([
        {
            "name":  m["name"],
            "prob":  round(m["prob"] * 100, 1),
            "color": m["color_hex"],
            "risk":  "HIGH"     if m["prob"] >= 0.5
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

    cam_y = -max_range * 2.2
    cam_z =  max_range * 0.5

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
  #header {{
    background: #10141c;
    border-bottom: 1px solid #1e2535;
    padding: 10px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-shrink: 0;
  }}
  #header h1 {{ font-size: 0.85rem; color: #00d4ff; letter-spacing: 0.08em; font-weight: 700; }}
  #header .overall {{ font-size: 1.1rem; font-weight: 700; color: {risk_color}; letter-spacing: 0.05em; }}
  #header .meta {{ font-size: 0.7rem; color: #64748b; }}
  #main {{ display: flex; flex: 1; overflow: hidden; }}
  #canvas-container {{ flex: 1; position: relative; }}
  #canvas-container canvas {{ display: block; width: 100% !important; height: 100% !important; }}
  #controls-hint {{
    position: absolute; bottom: 12px; left: 50%; transform: translateX(-50%);
    font-size: 0.65rem; color: #475569; letter-spacing: 0.05em; pointer-events: none;
  }}
  #sidebar {{
    width: 240px; background: #10141c; border-left: 1px solid #1e2535;
    overflow-y: auto; padding: 16px 12px; flex-shrink: 0;
  }}
  #sidebar h2 {{ font-size: 0.72rem; color: #64748b; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 14px; }}
  .region-item {{
    margin-bottom: 10px; padding: 8px 10px; background: #161b27;
    border-radius: 8px; border-left: 3px solid transparent;
    cursor: pointer; transition: background 0.15s;
  }}
  .region-item:hover {{ background: #1e2535; }}
  .region-name {{ font-size: 0.72rem; font-weight: 600; color: #e2e8f0; margin-bottom: 4px; }}
  .region-bar-wrap {{ background: #0d0d1a; border-radius: 4px; height: 6px; overflow: hidden; margin-bottom: 4px; }}
  .region-bar {{ height: 100%; border-radius: 4px; }}
  .region-prob {{ font-size: 0.65rem; color: #94a3b8; display: flex; justify-content: space-between; }}
  .risk-HIGH     {{ color: #e53935; }}
  .risk-ELEVATED {{ color: #fb8c00; }}
  .risk-LOW      {{ color: #43a047; }}
  #colorbar {{ margin-top: 20px; padding-top: 14px; border-top: 1px solid #1e2535; }}
  #colorbar h2 {{ font-size: 0.65rem; color: #64748b; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 8px; }}
  #colorbar-gradient {{
    height: 12px; border-radius: 6px;
    background: linear-gradient(to right, #1a237e, #00897b, #f9a825, #e65100, #b71c1c);
    margin-bottom: 4px;
  }}
  .cbar-labels {{ display: flex; justify-content: space-between; font-size: 0.6rem; color: #64748b; }}
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
    <div id="controls-hint">Drag to rotate &nbsp;&middot;&nbsp; Scroll to zoom &nbsp;&middot;&nbsp; Right-drag to pan</div>
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

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
(function() {{
  THREE.OrbitControls = function(camera, domElement) {{
    this.camera = camera;
    this.domElement = domElement;
    this.dampingFactor = 0.05;
    this.rotateSpeed = 0.8;
    this.zoomSpeed = 1.2;

    let isMouseDown = false, isRightDown = false;
    let lastX = 0, lastY = 0;
    const target = new THREE.Vector3({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f});
    const spherical = new THREE.Spherical();
    spherical.setFromVector3(camera.position.clone().sub(target));

    domElement.addEventListener('mousedown', e => {{
      isMouseDown = e.button === 0; isRightDown = e.button === 2;
      lastX = e.clientX; lastY = e.clientY;
    }});
    domElement.addEventListener('mouseup', () => {{ isMouseDown = false; isRightDown = false; }});
    domElement.addEventListener('contextmenu', e => e.preventDefault());
    domElement.addEventListener('mousemove', e => {{
      const dx = e.clientX - lastX, dy = e.clientY - lastY;
      lastX = e.clientX; lastY = e.clientY;
      if (isMouseDown) {{
        spherical.theta -= dx * 0.005 * this.rotateSpeed;
        spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, spherical.phi + dy * 0.005 * this.rotateSpeed));
      }}
      if (isRightDown) {{
        const pan = new THREE.Vector3(-dx * 0.3, dy * 0.3, 0);
        pan.applyQuaternion(camera.quaternion);
        target.add(pan);
      }}
    }});
    domElement.addEventListener('wheel', e => {{
      spherical.radius = Math.max(50, Math.min(800, spherical.radius + e.deltaY * 0.3 * this.zoomSpeed));
    }});
    this.update = function() {{
      camera.position.setFromSpherical(spherical).add(target);
      camera.lookAt(target);
    }};
  }};
}})();

const container = document.getElementById('canvas-container');
const scene     = new THREE.Scene();
scene.background = new THREE.Color(0x0d0d1a);
scene.fog        = new THREE.FogExp2(0x0d0d1a, 0.0008);

const camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 0.1, 2000);
camera.position.set(0, {cam_y:.1f}, {cam_z:.1f});
camera.lookAt({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f});

const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(container.clientWidth, container.clientHeight);
container.appendChild(renderer.domElement);

const ambient = new THREE.AmbientLight(0xffffff, 0.4); scene.add(ambient);
const key   = new THREE.DirectionalLight(0xffffff, 0.8); key.position.set(100, -200, 200); scene.add(key);
const fill  = new THREE.DirectionalLight(0x8899ff, 0.3); fill.position.set(-100, 100, -100); scene.add(fill);
const back  = new THREE.DirectionalLight(0xffeedd, 0.2); back.position.set(0, 200, -200); scene.add(back);

const regionMeshes = [];
{geometry_blocks}

const controls = new THREE.OrbitControls(camera, renderer.domElement);

function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}}
animate();

window.addEventListener('resize', () => {{
  camera.aspect = container.clientWidth / container.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(container.clientWidth, container.clientHeight);
}});

const legendData = {legend_items};
const legendEl   = document.getElementById('legend');
legendData.forEach(item => {{
  const div = document.createElement('div');
  div.className = 'region-item';
  div.style.borderLeftColor = item.color;
  div.innerHTML = `
    <div class="region-name">${{item.name}}</div>
    <div class="region-bar-wrap">
      <div class="region-bar" style="width:${{item.prob}}%;background:${{item.color}}"></div>
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
