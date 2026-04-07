import type { ImpactEvent, ImpactProfile, RiskLevel, HeadTrack } from '@/types/impact'

// ── Risk color palette ────────────────────────────────────────────────────────

const RISK_RGB: Record<RiskLevel, [number, number, number]> = {
  HIGH:     [239, 68,  68],
  ELEVATED: [245, 158, 11],
  LOW:      [34,  197, 94],
}

// ── Head tracking color palette ───────────────────────────────────────────────

const COLOR_IMPACT: [number, number, number] = [239, 68,  68]  // red-500
const COLOR_NORMAL: [number, number, number] = [56,  189, 248] // sky-400

// Per keypoint: nose, L-eye, R-eye, L-ear, R-ear
const KP_COLORS: [number, number, number][] = [
  [255, 255, 255],  // nose — white
  [147, 197, 253],  // L-eye — blue-200
  [147, 197, 253],  // R-eye
  [253, 224,  71],  // L-ear — yellow-300
  [253, 224,  71],  // R-ear
]
const KP_RADII = [3, 2.5, 2.5, 2.5, 2.5]

// ── Utility ───────────────────────────────────────────────────────────────────

function roundRectPath(
  ctx: CanvasRenderingContext2D,
  x: number, y: number, w: number, h: number, r: number,
) {
  ctx.beginPath()
  ctx.moveTo(x + r, y)
  ctx.lineTo(x + w - r, y)
  ctx.arcTo(x + w, y,     x + w, y + r,     r)
  ctx.lineTo(x + w, y + h - r)
  ctx.arcTo(x + w, y + h, x + w - r, y + h, r)
  ctx.lineTo(x + r, y + h)
  ctx.arcTo(x,     y + h, x,     y + h - r, r)
  ctx.lineTo(x,     y + r)
  ctx.arcTo(x,     y,     x + r, y,         r)
  ctx.closePath()
}

// ── Head tracking overlay ─────────────────────────────────────────────────────

function drawHeads(
  ctx: CanvasRenderingContext2D,
  heads: HeadTrack[],
  impactTracks: Set<number>,
  scaleX: number,
  scaleY: number,
) {
  for (const head of heads) {
    const { cx, cy, r, id, kp } = head
    const isImpact = impactTracks.has(id)
    const [cr, cg, cb] = isImpact ? COLOR_IMPACT : COLOR_NORMAL

    const sx = cx * scaleX
    const sy = cy * scaleY
    const sr = r  * scaleX  // uniform scale (video always letterboxed, not stretched)

    // Glow disk (radial gradient fill)
    const glow = ctx.createRadialGradient(sx, sy, sr * 0.5, sx, sy, sr * 1.4)
    glow.addColorStop(0,   `rgba(${cr},${cg},${cb},0.15)`)
    glow.addColorStop(0.5, `rgba(${cr},${cg},${cb},0.08)`)
    glow.addColorStop(1,   `rgba(${cr},${cg},${cb},0)`)
    ctx.beginPath()
    ctx.arc(sx, sy, sr * 1.4, 0, Math.PI * 2)
    ctx.fillStyle = glow
    ctx.fill()

    // Head ring
    ctx.beginPath()
    ctx.arc(sx, sy, sr, 0, Math.PI * 2)
    ctx.strokeStyle = `rgba(${cr},${cg},${cb},0.85)`
    ctx.lineWidth = isImpact ? 2.5 : 1.5
    ctx.stroke()

    // Keypoints
    kp.forEach((pt, i) => {
      if (pt === null) return
      const kx = pt[0] * scaleX
      const ky = pt[1] * scaleY
      const [kr, kg, kb] = KP_COLORS[i]
      const kr2 = KP_RADII[i]

      ctx.beginPath()
      ctx.arc(kx, ky, kr2, 0, Math.PI * 2)
      ctx.fillStyle = `rgb(${kr},${kg},${kb})`
      ctx.fill()

      ctx.strokeStyle = 'rgba(255,255,255,0.5)'
      ctx.lineWidth = 0.8
      ctx.stroke()
    })

    // Player ID badge
    const badgeText = `#${id}`
    ctx.font = 'bold 10px ui-monospace,monospace'
    const textW = ctx.measureText(badgeText).width
    const pillW = textW + 10
    const pillH = 16
    const pillX = sx - pillW / 2
    const pillY = sy - sr - 12 - pillH

    roundRectPath(ctx, pillX, pillY, pillW, pillH, 4)
    ctx.fillStyle = `rgba(${cr},${cg},${cb},0.30)`
    ctx.fill()

    ctx.fillStyle = `rgb(${cr},${cg},${cb})`
    ctx.textAlign = 'center'
    ctx.fillText(badgeText, sx, pillY + pillH - 4)
    ctx.textAlign = 'left'
  }
}

// ── Edge flash ────────────────────────────────────────────────────────────────

function drawEdgeFlash(
  ctx: CanvasRenderingContext2D,
  W: number, H: number,
  ageMs: number,
) {
  const opacity = Math.max(0, 1 - ageMs / 1000)
  if (opacity === 0) return

  // White pop flash on very first frames
  const popAlpha = Math.max(0, 0.3 - ageMs / 333)
  if (popAlpha > 0) {
    ctx.fillStyle = `rgba(255,255,255,${popAlpha})`
    ctx.fillRect(0, 0, W, H)
  }

  const depth = Math.min(W, H) * 0.28
  const alpha = opacity * 0.85

  const top = ctx.createLinearGradient(0, 0, 0, depth)
  top.addColorStop(0, `rgba(220,38,38,${alpha})`)
  top.addColorStop(1, 'rgba(220,38,38,0)')
  ctx.fillStyle = top
  ctx.fillRect(0, 0, W, depth)

  const bot = ctx.createLinearGradient(0, H, 0, H - depth)
  bot.addColorStop(0, `rgba(220,38,38,${alpha})`)
  bot.addColorStop(1, 'rgba(220,38,38,0)')
  ctx.fillStyle = bot
  ctx.fillRect(0, H - depth, W, depth)

  const left = ctx.createLinearGradient(0, 0, depth, 0)
  left.addColorStop(0, `rgba(220,38,38,${alpha})`)
  left.addColorStop(1, 'rgba(220,38,38,0)')
  ctx.fillStyle = left
  ctx.fillRect(0, 0, depth, H)

  const right = ctx.createLinearGradient(W, 0, W - depth, 0)
  right.addColorStop(0, `rgba(220,38,38,${alpha})`)
  right.addColorStop(1, 'rgba(220,38,38,0)')
  ctx.fillStyle = right
  ctx.fillRect(W - depth, 0, depth, H)
}

// ── Impact HUD panel ──────────────────────────────────────────────────────────

function metricPill(
  ctx: CanvasRenderingContext2D,
  x: number, y: number,
  label: string, value: string,
  risk: RiskLevel, alpha: number,
) {
  const [r, g, b] = RISK_RGB[risk]
  const PW = 62, PH = 34

  roundRectPath(ctx, x, y, PW, PH, 4)
  ctx.fillStyle = `rgba(${r},${g},${b},${0.18 * alpha})`
  ctx.fill()
  ctx.strokeStyle = `rgba(${r},${g},${b},${0.55 * alpha})`
  ctx.lineWidth = 1
  ctx.stroke()

  ctx.fillStyle = `rgba(148,163,184,${alpha})`
  ctx.font = `8px ui-sans-serif,system-ui,sans-serif`
  ctx.fillText(label, x + 5, y + 12)

  ctx.fillStyle = `rgba(${r},${g},${b},${alpha})`
  ctx.font = `bold 11px ui-monospace,monospace`
  ctx.fillText(value, x + 5, y + 26)
}

function drawHUD(
  ctx: CanvasRenderingContext2D,
  W: number, H: number,
  event: ImpactEvent,
  ageMs: number,
  profiles: ImpactProfile[],
) {
  const panelAlpha = Math.min(1, ageMs / 150)
  if (panelAlpha === 0) return

  const evtProfiles = profiles.filter(
    p => p.event_frame === event.frame && event.tracks.includes(p.track_id),
  )

  const PAD = 16
  const PANEL_W = Math.max(200, Math.min(260, W * 0.38))
  const HEADER_H = 52
  const ROW_H = 60
  const FOOTER_H = evtProfiles.length > 0 ? 36 : 0
  const PANEL_H = HEADER_H + evtProfiles.length * ROW_H + FOOTER_H + 12

  const PANEL_X = W - PAD - PANEL_W
  const PANEL_Y = PAD

  roundRectPath(ctx, PANEL_X, PANEL_Y, PANEL_W, PANEL_H, 8)
  ctx.fillStyle = `rgba(15,23,42,${0.90 * panelAlpha})`
  ctx.fill()
  ctx.strokeStyle = `rgba(239,68,68,${0.55 * panelAlpha})`
  ctx.lineWidth = 1.5
  ctx.stroke()

  ctx.fillStyle = `rgba(239,68,68,${panelAlpha})`
  ctx.font = `bold 11px ui-monospace,monospace`
  ctx.fillText('\u26A1 IMPACT DETECTED', PANEL_X + 12, PANEL_Y + 18)

  ctx.fillStyle = `rgba(148,163,184,${panelAlpha})`
  ctx.font = `10px ui-sans-serif,system-ui,sans-serif`
  const conf = Math.round(event.confidence * 100)
  const players = event.tracks.map(t => `#${t}`).join(', ')
  ctx.fillText(`Conf: ${conf}%  ·  Players: ${players}`, PANEL_X + 12, PANEL_Y + 34)

  ctx.strokeStyle = `rgba(51,65,85,${panelAlpha})`
  ctx.lineWidth = 1
  ctx.beginPath()
  ctx.moveTo(PANEL_X + 12, PANEL_Y + 42)
  ctx.lineTo(PANEL_X + PANEL_W - 12, PANEL_Y + 42)
  ctx.stroke()

  evtProfiles.forEach((p, i) => {
    const rowY = PANEL_Y + HEADER_H + i * ROW_H

    ctx.fillStyle = `rgba(226,232,240,${panelAlpha})`
    ctx.font = `bold 10px ui-sans-serif,system-ui,sans-serif`
    ctx.fillText(`Player #${p.track_id}`, PANEL_X + 12, rowY + 14)

    const pillGap = 4
    const totalPillW = 3 * 62 + 2 * pillGap
    const pillStartX = PANEL_X + (PANEL_W - totalPillW) / 2

    metricPill(ctx, pillStartX,          rowY + 20, 'BrIC_R', p.bric_r.toFixed(3),        p.bric_r_risk, panelAlpha)
    metricPill(ctx, pillStartX + 66,     rowY + 20, 'KLC',    p.klc_rot_rad_s.toFixed(1),  p.klc_risk,    panelAlpha)
    metricPill(ctx, pillStartX + 66 * 2, rowY + 20, 'DMGE',   p.damage.toFixed(3),         p.damage_risk, panelAlpha)
  })

  if (evtProfiles.length > 0) {
    const worstRisk = evtProfiles.reduce<RiskLevel>((acc, p) => {
      const order: Record<RiskLevel, number> = { LOW: 0, ELEVATED: 1, HIGH: 2 }
      return order[p.risk_summary] > order[acc] ? p.risk_summary : acc
    }, 'LOW')

    const [r, g, b] = RISK_RGB[worstRisk]
    const badgeY = PANEL_Y + HEADER_H + evtProfiles.length * ROW_H + 4
    const badgeW = 90, badgeH = 22
    const badgeX = PANEL_X + (PANEL_W - badgeW) / 2

    roundRectPath(ctx, badgeX, badgeY, badgeW, badgeH, 4)
    ctx.fillStyle = `rgba(${r},${g},${b},${0.2 * panelAlpha})`
    ctx.fill()
    ctx.strokeStyle = `rgba(${r},${g},${b},${0.6 * panelAlpha})`
    ctx.lineWidth = 1
    ctx.stroke()

    ctx.fillStyle = `rgba(${r},${g},${b},${panelAlpha})`
    ctx.font = `bold 10px ui-monospace,monospace`
    ctx.textAlign = 'center'
    ctx.fillText(`RISK: ${worstRisk}`, badgeX + badgeW / 2, badgeY + 15)
    ctx.textAlign = 'left'
  }
}

// ── Main export ───────────────────────────────────────────────────────────────

export function drawOverlay(
  canvas: HTMLCanvasElement | null,
  event: ImpactEvent | null,
  freezeStartMs: number,
  profiles: ImpactProfile[],
  tracking: Map<number, HeadTrack[]> | null,
  currentFrame: number,
  impactTracks: Set<number>,
  scaleX: number,
  scaleY: number,
  showHud: boolean,
): void {
  if (!canvas) return
  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const W = canvas.width
  const H = canvas.height
  ctx.clearRect(0, 0, W, H)

  // Head tracking overlays (only present for impact-window frames)
  if (tracking) {
    const heads = tracking.get(currentFrame)
    if (heads && heads.length > 0) {
      drawHeads(ctx, heads, impactTracks, scaleX, scaleY)
    }
  }

  // Impact HUD panel (draws on top of head overlays; gated by server config)
  if (!event || !showHud) return
  const ageMs = Date.now() - freezeStartMs
  drawHUD(ctx, W, H, event, ageMs, profiles)
}
