/**
 * BrainVisualization.tsx
 *
 * Animated 3D rotating brain (Harvard-Oxford atlas meshes) with a pulsing
 * impact hotspot derived from the peak angular velocity direction.
 *
 * Brain is rendered in neutral anatomical colours; only the hotspot marks
 * the region of rotational strain.
 *
 * MNI coordinate remapping so brainstem faces down:
 *   MNI X → Three.js X  (left–right, unchanged)
 *   MNI Z → Three.js Y  (inferior–superior → down–up)
 *   MNI Y → Three.js Z  (posterior–anterior → depth)
 * Implemented by rotating the brain group by +π/2 around X.
 */

import { useEffect, useRef, useState } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import * as THREE from 'three'
import type { RegionalTBIProbs, BrainRegionMesh } from '@/types/impact'

// ── Module-level mesh cache ────────────────────────────────────────────────
let _meshCache: BrainRegionMesh[] | null = null
let _meshPromise: Promise<BrainRegionMesh[]> | null = null

function loadMeshData(): Promise<BrainRegionMesh[]> {
  if (_meshCache) return Promise.resolve(_meshCache)
  if (_meshPromise) return _meshPromise
  _meshPromise = fetch('/brain_regions.json')
    .then(r => r.json())
    .then((data: BrainRegionMesh[]) => { _meshCache = data; return data })
  return _meshPromise
}

// ── Anatomical colour palette (neutral, no TBI mapping) ────────────────────
const REGION_COLOR: Record<string, string> = {
  grey_matter:     '#c4a898',   // warm pinkish-grey — cortex
  white_matter:    '#ddd5cc',   // cream — white matter
  corpus_callosum: '#b8b0cc',   // soft lavender-grey
  thalamus:        '#8898a8',   // cool blue-grey
  brainstem:       '#909888',   // olive-grey
  cerebellum:      '#b8a0ac',   // dusty rose-grey
}

// ── Per-region render config ───────────────────────────────────────────────
interface RegionCfg {
  opacity: number
  transparent: boolean
  depthWrite: boolean
  scale: number
  renderOrder: number
  side: THREE.Side
}

const REGION_CFG: Record<string, RegionCfg> = {
  grey_matter:     { opacity: 0.30, transparent: true,  depthWrite: false, scale: 1.000, renderOrder: 0, side: THREE.FrontSide  },
  white_matter:    { opacity: 0.50, transparent: true,  depthWrite: false, scale: 0.980, renderOrder: 1, side: THREE.FrontSide  },
  corpus_callosum: { opacity: 0.65, transparent: true,  depthWrite: true,  scale: 0.960, renderOrder: 2, side: THREE.DoubleSide },
  thalamus:        { opacity: 0.90, transparent: false, depthWrite: true,  scale: 1.000, renderOrder: 3, side: THREE.FrontSide  },
  brainstem:       { opacity: 0.90, transparent: false, depthWrite: true,  scale: 1.000, renderOrder: 4, side: THREE.FrontSide  },
  cerebellum:      { opacity: 0.90, transparent: false, depthWrite: true,  scale: 1.000, renderOrder: 5, side: THREE.FrontSide  },
}

// ── Impact direction ───────────────────────────────────────────────────────
// cross(omega_peak, up=[0,1,0]) gives a vector perpendicular to the rotation
// axis — the direction of the rotational impulse on the brain surface.
function computeImpactDir(
  omegaMagnitudes: number[],
  omegaUnitVectors: number[][],
): THREE.Vector3 | null {
  if (!omegaMagnitudes.length || !omegaUnitVectors.length) return null
  const peakIdx = omegaMagnitudes.reduce(
    (best, v, i) => (v > omegaMagnitudes[best] ? i : best), 0
  )
  const [ox, oy, oz] = omegaUnitVectors[peakIdx]
  const dir = new THREE.Vector3(-oz, 0, ox)   // cross([ox,oy,oz], [0,1,0])
  if (dir.lengthSq() < 1e-6) return null
  dir.y = oy * 0.4   // blend in vertical component so top/bottom impacts register
  return dir.normalize()
}


// ── Geometry builder ───────────────────────────────────────────────────────
function buildGeometry(
  region: BrainRegionMesh,
  centerOffset: THREE.Vector3,
  normScale: number,
): THREE.BufferGeometry {
  const flat = new Float32Array(region.vertices.length * 3)
  for (let i = 0; i < region.vertices.length; i++) {
    flat[i * 3 + 0] = (region.vertices[i][0] - centerOffset.x) * normScale
    flat[i * 3 + 1] = (region.vertices[i][1] - centerOffset.y) * normScale
    flat[i * 3 + 2] = (region.vertices[i][2] - centerOffset.z) * normScale
  }
  const idx = new Uint32Array(region.faces.length * 3)
  for (let i = 0; i < region.faces.length; i++) {
    idx[i * 3 + 0] = region.faces[i][0]
    idx[i * 3 + 1] = region.faces[i][1]
    idx[i * 3 + 2] = region.faces[i][2]
  }
  const geo = new THREE.BufferGeometry()
  geo.setAttribute('position', new THREE.BufferAttribute(flat, 3))
  geo.setIndex(new THREE.BufferAttribute(idx, 1))
  geo.computeVertexNormals()
  return geo
}

// ── Brain scene ────────────────────────────────────────────────────────────
interface BrainSceneProps {
  meshData:  BrainRegionMesh[]
  impactDir: THREE.Vector3 | null
}

function BrainScene({ meshData, impactDir }: BrainSceneProps) {
  const groupRef  = useRef<THREE.Group>(null!)
  const meshesRef = useRef<{ geo: THREE.BufferGeometry; mat: THREE.MeshStandardMaterial }[]>([])

  // Centroid + normalisation scale derived from grey_matter (largest mesh)
  const { centerOffset, normScale } = (() => {
    const gm = meshData.find(m => m.key === 'grey_matter') ?? meshData[0]
    let sx = 0, sy = 0, sz = 0
    let minX = Infinity, maxX = -Infinity
    let minY = Infinity, maxY = -Infinity
    let minZ = Infinity, maxZ = -Infinity
    for (const [x, y, z] of gm.vertices) {
      sx += x; sy += y; sz += z
      if (x < minX) minX = x; if (x > maxX) maxX = x
      if (y < minY) minY = y; if (y > maxY) maxY = y
      if (z < minZ) minZ = z; if (z > maxZ) maxZ = z
    }
    const n = gm.vertices.length
    const center = new THREE.Vector3(sx / n, sy / n, sz / n)
    const span   = Math.max(maxX - minX, maxY - minY, maxZ - minZ)
    return { centerOffset: center, normScale: 1.6 / span }
  })()

  useEffect(() => {
    const built: { geo: THREE.BufferGeometry; mat: THREE.MeshStandardMaterial }[] = []

    const sorted = [...meshData].sort(
      (a, b) => (REGION_CFG[a.key]?.renderOrder ?? 99) - (REGION_CFG[b.key]?.renderOrder ?? 99)
    )

    for (const region of sorted) {
      const cfg = REGION_CFG[region.key]
      if (!cfg) continue

      const geo = buildGeometry(region, centerOffset, normScale)
      const mat = new THREE.MeshStandardMaterial({
        color:       new THREE.Color(REGION_COLOR[region.key] ?? '#a0a0a0'),
        roughness:   0.70,
        metalness:   0.05,
        opacity:     cfg.opacity,
        transparent: cfg.transparent,
        depthWrite:  cfg.depthWrite,
        side:        cfg.side,
      })

      const mesh = new THREE.Mesh(geo, mat)
      mesh.renderOrder = cfg.renderOrder
      mesh.scale.setScalar(cfg.scale)
      groupRef.current.add(mesh)
      built.push({ geo, mat })
    }

    meshesRef.current = built

    return () => {
      for (const { geo, mat } of meshesRef.current) {
        geo.dispose()
        mat.dispose()
      }
      while (groupRef.current.children.length > 0) {
        groupRef.current.remove(groupRef.current.children[0])
      }
    }
  }, [meshData, centerOffset, normScale])   // intentionally excludes impactDir — hotspot is JSX

  // Rotation: MNI Z (superior) → Three.js Y (up), so brainstem faces down
  return (
    <>
      <color attach="background" args={['#f8fafc']} />
      <ambientLight intensity={1.4} />
      <directionalLight position={[2, 3, 2]}   intensity={1.6} />
      <directionalLight position={[-2, -1, -2]} intensity={0.5} color="#dde4ed" />
      <group ref={groupRef} rotation={[-Math.PI / 2, 0, 0]}>
        {impactDir && (
          <pointLight
            position={impactDir.clone().multiplyScalar(0.75).toArray() as [number,number,number]}
            color="#ff2200"
            intensity={4}
            distance={0.9}
            decay={2}
          />
        )}
      </group>
    </>
  )
}

// ── Loading skeleton ───────────────────────────────────────────────────────
function BrainSkeleton() {
  return (
    <div className="aspect-square w-[80%] max-w-[375px] mx-auto bg-slate-50 flex items-center justify-center">
      <div className="flex flex-col items-center gap-2">
        <div className="w-16 h-16 rounded-full border border-slate-300 border-t-blue-400 animate-spin" />
        <span className="text-[10px] text-slate-400">Loading brain model…</span>
      </div>
    </div>
  )
}

// ── Public component ───────────────────────────────────────────────────────
interface Props {
  regionalProbs?:     RegionalTBIProbs
  omegaMagnitudes?:   number[]
  omegaUnitVectors?:  number[][]
}

export default function BrainVisualization({ omegaMagnitudes, omegaUnitVectors }: Props) {
  const [meshData, setMeshData] = useState<BrainRegionMesh[] | null>(null)
  const [error, setError]       = useState(false)

  useEffect(() => {
    loadMeshData().then(setMeshData).catch(() => setError(true))
  }, [])

  const impactDir = (omegaMagnitudes && omegaUnitVectors)
    ? computeImpactDir(omegaMagnitudes, omegaUnitVectors)
    : null

  if (error) return (
    <div className="aspect-square w-[80%] max-w-[375px] mx-auto bg-slate-50 flex items-center justify-center">
      <span className="text-[10px] text-slate-400">Brain model unavailable</span>
    </div>
  )
  if (!meshData) return <BrainSkeleton />

  return (
    <div className="aspect-square w-[80%] max-w-[375px] mx-auto bg-slate-50 relative rounded-b-xl overflow-hidden">
      <Canvas
        camera={{ position: [0, 0.15, 3.4], fov: 38 }}
        gl={{ antialias: true, alpha: false }}
        style={{ background: '#f8fafc' }}
      >
        <BrainScene meshData={meshData} impactDir={impactDir} />
        <OrbitControls
          autoRotate
          autoRotateSpeed={0.7}
          enableZoom={false}
          enablePan={false}
          minPolarAngle={Math.PI * 0.20}
          maxPolarAngle={Math.PI * 0.80}
        />
      </Canvas>

    </div>
  )
}
