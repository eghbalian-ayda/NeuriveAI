import { useRef, useState } from 'react'
import * as Tooltip from '@radix-ui/react-tooltip'
import type { ImpactEvent, ImpactProfile } from '@/types/impact'
import { frameToTimecode, markerColor, worstRisk } from '@/lib/utils'

interface MarkerInfo {
  event: ImpactEvent
  profiles: ImpactProfile[]
  leftPct: number
  height: number
  color: string
}

interface Props {
  events: ImpactEvent[]
  profiles: ImpactProfile[]
  totalFrames: number
  fps: number
  playheadPct: number
  onSeek: (frame: number) => void
  onMarkerClick?: (trackIds: number[]) => void
}

export default function ImpactTimeline({
  events,
  profiles,
  totalFrames,
  fps,
  playheadPct,
  onSeek,
  onMarkerClick,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null)

  const markers: MarkerInfo[] = events.map((ev) => {
    const evProfiles = profiles.filter(p => p.event_frame === ev.frame && ev.tracks.includes(p.track_id))
    const risk = evProfiles.length > 0
      ? worstRisk(evProfiles.map(p => p.risk_summary))
      : 'LOW'
    return {
      event: ev,
      profiles: evProfiles,
      leftPct: totalFrames > 0 ? (ev.frame / totalFrames) * 100 : 0,
      height: Math.round(16 + ev.confidence * 32),
      color: markerColor(risk),
    }
  })

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
          className="relative h-14 bg-slate-900 rounded-xl cursor-pointer border border-slate-800 overflow-hidden"
          onClick={handleBarClick}
        >
          {/* Track line */}
          <div className="absolute inset-x-0 bottom-3 h-px bg-slate-700" />

          {/* Impact markers */}
          {markers.map((m, i) => (
            <Tooltip.Root key={i} open={hoveredIdx === i}>
              <Tooltip.Trigger asChild>
                <div
                  className="absolute bottom-0 w-1.5 rounded-t-sm transition-opacity cursor-pointer"
                  style={{
                    left: `${m.leftPct}%`,
                    height: `${m.height}px`,
                    backgroundColor: m.color,
                    transform: 'translateX(-50%)',
                    opacity: hoveredIdx !== null && hoveredIdx !== i ? 0.5 : 1,
                  }}
                  onMouseEnter={() => setHoveredIdx(i)}
                  onMouseLeave={() => setHoveredIdx(null)}
                  onClick={(e) => {
                    e.stopPropagation()
                    onSeek(m.event.frame)
                    onMarkerClick?.(m.event.tracks)
                  }}
                />
              </Tooltip.Trigger>
              <Tooltip.Portal>
                <Tooltip.Content
                  className="
                    z-50 bg-slate-800 border border-slate-700 rounded-lg px-3 py-2
                    text-xs text-slate-200 shadow-xl animate-fade-in
                  "
                  side="top"
                  sideOffset={6}
                >
                  <div className="font-semibold">{frameToTimecode(m.event.frame, fps)}</div>
                  <div className="text-slate-400">
                    Frame {m.event.frame} · {Math.round(m.event.confidence * 100)}% conf
                  </div>
                  <div className="text-slate-400">Tracks: {m.event.tracks.join(', ')}</div>
                  <div className="flex gap-1 mt-1 flex-wrap">
                    {m.event.stages.map(s => (
                      <span key={s} className="bg-slate-700 rounded px-1 uppercase text-slate-300 text-[10px]">
                        {s}
                      </span>
                    ))}
                  </div>
                  <Tooltip.Arrow className="fill-slate-800" />
                </Tooltip.Content>
              </Tooltip.Portal>
            </Tooltip.Root>
          ))}

          {/* Playhead */}
          <div
            className="absolute top-0 bottom-0 w-0.5 bg-white/80 pointer-events-none"
            style={{ left: `${playheadPct}%` }}
          >
            <div className="w-2.5 h-2.5 rounded-full bg-white absolute -top-1 -translate-x-1/2" />
          </div>
        </div>

        {/* Time labels */}
        <div className="flex justify-between mt-1 px-0.5">
          {[0, 0.25, 0.5, 0.75, 1].map((pct) => (
            <span key={pct} className="text-slate-600 text-[10px] font-mono">
              {frameToTimecode(Math.round(totalFrames * pct), fps)}
            </span>
          ))}
        </div>
      </div>
    </Tooltip.Provider>
  )
}
