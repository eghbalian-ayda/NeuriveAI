import { useRef, useState, useEffect } from 'react'
import * as Tooltip from '@radix-ui/react-tooltip'
import type { ImpactEvent, ImpactProfile } from '@/types/impact'
import { frameToTimecode, markerColor, worstRisk } from '@/lib/utils'

interface WindowInfo {
  event: ImpactEvent
  risk: ReturnType<typeof worstRisk>
  color: string
  impactPct: number
  windowStartPct: number
  windowEndPct: number
  height: number
  profiles: ImpactProfile[]
}

interface Props {
  events: ImpactEvent[]
  profiles: ImpactProfile[]
  totalFrames: number
  fps: number
  playheadPct: number
  videoSrc: string
  onSeek: (frame: number) => void
  onMarkerClick?: (trackIds: number[]) => void
}

function hexToRgba(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16)
  const g = parseInt(hex.slice(3, 5), 16)
  const b = parseInt(hex.slice(5, 7), 16)
  return `rgba(${r},${g},${b},${alpha})`
}

export default function ImpactTimeline({
  events,
  profiles,
  totalFrames,
  fps,
  playheadPct,
  videoSrc,
  onSeek,
  onMarkerClick,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null)

  // Build per-event window info
  const windows: WindowInfo[] = events.map((ev) => {
    const evProfiles = profiles.filter(
      (p) => p.event_frame === ev.frame && ev.tracks.includes(p.track_id),
    )
    const risk = evProfiles.length > 0
      ? worstRisk(evProfiles.map((p) => p.risk_summary))
      : 'LOW'

    // Window extent from profile frame_indices, or fall back to ±15 frames
    let windowStart = ev.frame - 15
    let windowEnd = ev.frame + 15
    if (evProfiles.length > 0) {
      const allIdx = evProfiles.flatMap((p) => p.frame_indices)
      if (allIdx.length > 0) {
        windowStart = Math.min(...allIdx)
        windowEnd = Math.max(...allIdx)
      }
    }

    const toP = (f: number) => totalFrames > 0 ? (Math.max(0, Math.min(totalFrames, f)) / totalFrames) * 100 : 0

    return {
      event: ev,
      risk,
      color: markerColor(risk),
      impactPct: toP(ev.frame),
      windowStartPct: toP(windowStart),
      windowEndPct: toP(windowEnd),
      height: Math.round(16 + ev.confidence * 32),
      profiles: evProfiles,
    }
  })

  // Draw filmstrip onto canvas
  useEffect(() => {
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container || !videoSrc) return

    canvas.width = container.offsetWidth
    canvas.height = container.offsetHeight

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    ctx.clearRect(0, 0, canvas.width, canvas.height)

    const vid = document.createElement('video')
    vid.src = videoSrc
    vid.muted = true
    vid.preload = 'metadata'
    vid.crossOrigin = 'anonymous'

    const N = Math.max(10, Math.floor(canvas.width / 36))
    const frameW = canvas.width / N
    let i = 0
    let cancelled = false

    function drawFrame() {
      if (cancelled || !ctx || !canvas) return
      ctx.drawImage(vid, i * frameW, 0, frameW, canvas.height)
      i++
      if (i < N) vid.currentTime = (i / N) * vid.duration
    }

    vid.addEventListener('loadedmetadata', () => {
      if (cancelled) return
      i = 0
      vid.currentTime = 0
    })
    vid.addEventListener('seeked', drawFrame)

    return () => {
      cancelled = true
      vid.removeEventListener('seeked', drawFrame)
      vid.src = ''
    }
  }, [videoSrc])

  function handleBarClick(e: React.MouseEvent<HTMLDivElement>) {
    if (!containerRef.current) return
    const rect = containerRef.current.getBoundingClientRect()
    const pct = (e.clientX - rect.left) / rect.width
    const frame = Math.round(pct * totalFrames)
    onSeek(Math.max(0, Math.min(frame, totalFrames)))
  }

  return (
    <Tooltip.Provider delayDuration={150}>
      <div className="mt-2 px-1">
        {/* Timeline bar */}
        <div
          ref={containerRef}
          className="relative h-14 rounded-xl cursor-pointer border border-slate-200 overflow-hidden bg-slate-900"
          onClick={handleBarClick}
        >
          {/* Filmstrip canvas */}
          <canvas
            ref={canvasRef}
            className="absolute inset-0 w-full h-full"
            style={{ display: 'block' }}
          />

          {/* Scrim so markers/text stay legible */}
          <div className="absolute inset-0 bg-black/30" />

          {/* Impact windows */}
          {windows.map((w, i) => (
            <div
              key={i}
              className="absolute top-0 bottom-0 pointer-events-none"
              style={{
                left: `${w.windowStartPct}%`,
                width: `${w.windowEndPct - w.windowStartPct}%`,
                background: hexToRgba(w.color, 0.18),
                borderLeft: `1.5px solid ${hexToRgba(w.color, 0.7)}`,
                borderRight: `1.5px solid ${hexToRgba(w.color, 0.7)}`,
              }}
            />
          ))}

          {/* Track line */}
          <div className="absolute inset-x-0 bottom-3 h-px bg-white/30" />

          {/* Impact markers */}
          {windows.map((w, i) => (
            <Tooltip.Root key={i} open={hoveredIdx === i}>
              <Tooltip.Trigger asChild>
                <div
                  className="absolute bottom-0 w-1.5 rounded-t-sm transition-opacity cursor-pointer"
                  style={{
                    left: `${w.impactPct}%`,
                    height: `${w.height}px`,
                    backgroundColor: w.color,
                    transform: 'translateX(-50%)',
                    opacity: hoveredIdx !== null && hoveredIdx !== i ? 0.5 : 1,
                  }}
                  onMouseEnter={() => setHoveredIdx(i)}
                  onMouseLeave={() => setHoveredIdx(null)}
                  onClick={(e) => {
                    e.stopPropagation()
                    onSeek(w.event.frame)
                    onMarkerClick?.(w.event.tracks)
                  }}
                />
              </Tooltip.Trigger>
              <Tooltip.Portal>
                <Tooltip.Content
                  className="
                    z-50 bg-white border border-slate-200 rounded-lg px-3 py-2
                    text-xs text-slate-800 shadow-lg animate-fade-in
                  "
                  side="top"
                  sideOffset={6}
                >
                  <div className="font-semibold">{frameToTimecode(w.event.frame, fps)}</div>
                  <div className="text-slate-500">
                    Frame {w.event.frame} · {Math.round(w.event.confidence * 100)}% conf
                  </div>
                  <div className="text-slate-500">Tracks: {w.event.tracks.join(', ')}</div>
                  <div className="flex gap-1 mt-1 flex-wrap">
                    {w.event.stages.map((s) => (
                      <span key={s} className="bg-slate-100 rounded px-1 uppercase text-slate-600 text-[10px]">
                        {s}
                      </span>
                    ))}
                  </div>
                  <Tooltip.Arrow className="fill-white" style={{ filter: 'drop-shadow(0 1px 1px rgba(0,0,0,0.1))' }} />
                </Tooltip.Content>
              </Tooltip.Portal>
            </Tooltip.Root>
          ))}

          {/* Playhead */}
          <div
            className="absolute top-0 bottom-0 w-0.5 bg-white/90 pointer-events-none"
            style={{ left: `${playheadPct}%` }}
          >
            <div className="w-2.5 h-2.5 rounded-full bg-white absolute -top-1 -translate-x-1/2" />
          </div>
        </div>

        {/* Time labels */}
        <div className="flex justify-between mt-1 px-0.5">
          {[0, 0.25, 0.5, 0.75, 1].map((pct) => (
            <span key={pct} className="text-slate-400 text-[10px] font-mono">
              {frameToTimecode(Math.round(totalFrames * pct), fps)}
            </span>
          ))}
        </div>
      </div>
    </Tooltip.Provider>
  )
}
