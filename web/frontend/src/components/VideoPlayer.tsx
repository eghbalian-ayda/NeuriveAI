import { useEffect, useRef, useState, useCallback, useMemo, forwardRef, useImperativeHandle } from 'react'
import { Play, Pause, Maximize2, ChevronLeft, ChevronRight, ZoomIn, PauseCircle } from 'lucide-react'
import type { ImpactEvent, ImpactProfile, HeadTrack, TrackingMap } from '@/types/impact'
import ImpactTimeline from './ImpactTimeline'
import { drawOverlay } from '@/lib/canvasAnnotations'

export interface VideoPlayerHandle {
  seekToFrame: (frame: number) => void
}

interface Props {
  jobId: string
  events: ImpactEvent[]
  profiles: ImpactProfile[]
  totalFrames: number
  fps: number
  onMarkerClick?: (trackIds: number[]) => void
  tracking?: TrackingMap
}

const VideoPlayer = forwardRef<VideoPlayerHandle, Props>(function VideoPlayer(
  { jobId, events, profiles, totalFrames, fps, onMarkerClick, tracking },
  ref,
) {
  const videoRef         = useRef<HTMLVideoElement>(null)
  const canvasRef        = useRef<HTMLCanvasElement>(null)
  const videoContainerRef = useRef<HTMLDivElement>(null)   // for outer glow box-shadow
  const [playheadPct, setPlayheadPct] = useState(0)
  const [isFrozen, setIsFrozen] = useState(false)          // true only when Pause toggle fires
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const rafRef = useRef<number>(0)

  // Control toggles
  const [zoomEnabled,   setZoomEnabled]   = useState(false)
  const [pauseEnabled,  setPauseEnabled]  = useState(false)  // off by default
  const [zoomTransform, setZoomTransform] = useState({ scale: 1, tx: 0, ty: 0 })
  const [showImpactHud, setShowImpactHud] = useState(false)  // populated from /api/config

  // Impact-frame detection
  const freezeEventRef    = useRef<ImpactEvent | null>(null)
  const freezeStartMsRef  = useRef<number>(0)
  const prevFrameRef      = useRef<number>(-1)    // previous RAF frame; used for crossing detection
  const flashStartMsRef   = useRef<number>(0)     // timestamp of last impact crossing (for outer glow)

  // Frame→event lookup; rebuilt whenever events changes
  const eventByFrameRef = useRef(new Map<number, ImpactEvent>())
  useEffect(() => {
    const m = new Map<number, ImpactEvent>()
    for (const ev of events) m.set(ev.frame, ev)
    eventByFrameRef.current = m
  }, [events])

  // Keep profiles accessible inside the RAF closure without recreating it
  const profilesRef = useRef(profiles)
  useEffect(() => { profilesRef.current = profiles }, [profiles])

  // Convert TrackingMap (string-keyed) to Map<number, HeadTrack[]> once
  const trackingMap = useMemo<Map<number, HeadTrack[]>>(() => {
    const m = new Map<number, HeadTrack[]>()
    if (!tracking) return m
    for (const [key, heads] of Object.entries(tracking)) m.set(Number(key), heads)
    return m
  }, [tracking])

  const trackingRef = useRef<Map<number, HeadTrack[]>>(new Map())
  useEffect(() => { trackingRef.current = trackingMap }, [trackingMap])

  // Rendered video size — updated by ResizeObserver
  const renderedSizeRef = useRef({ w: 0, h: 0 })

  // RAF-accessible mirrors of toggle state (avoid recreating updatePlayhead on each toggle)
  const zoomEnabledRef   = useRef(false)
  const pauseEnabledRef  = useRef(false)
  const showImpactHudRef = useRef(false)
  useEffect(() => { zoomEnabledRef.current   = zoomEnabled   }, [zoomEnabled])
  useEffect(() => { pauseEnabledRef.current  = pauseEnabled  }, [pauseEnabled])
  useEffect(() => { showImpactHudRef.current = showImpactHud }, [showImpactHud])

  // Fetch HUD visibility config from backend once on mount
  useEffect(() => {
    fetch('/api/config')
      .then(r => r.json())
      .then(cfg => setShowImpactHud(cfg.showImpactHud ?? false))
      .catch(() => { /* stay at default false */ })
  }, [])

  // Which zoom window the playhead is currently inside (null = none)
  const activeWindowRef = useRef<{ eventFrame: number } | null>(null)

  // Precomputed per-event window bounds (mirrors ImpactTimeline's WindowInfo logic)
  const impactWindowsRef = useRef<{ eventFrame: number; windowStart: number; windowEnd: number }[]>([])
  useEffect(() => {
    impactWindowsRef.current = events.map(ev => {
      const evProfiles = profiles.filter(
        p => p.event_frame === ev.frame && ev.tracks.includes(p.track_id),
      )
      let windowStart = ev.frame - 15
      let windowEnd   = ev.frame + 15
      if (evProfiles.length > 0) {
        const allIdx = evProfiles.flatMap(p => p.frame_indices)
        if (allIdx.length > 0) {
          windowStart = Math.min(...allIdx)
          windowEnd   = Math.max(...allIdx)
        }
      }
      return { eventFrame: ev.frame, windowStart, windowEnd }
    })
  }, [events, profiles])

  // Sorted events for prev/next navigation
  const eventsSortedRef = useRef<ImpactEvent[]>([])
  useEffect(() => {
    eventsSortedRef.current = [...events].sort((a, b) => a.frame - b.frame)
  }, [events])

  // Compute a CSS zoom transform centred on the impacting heads.
  // Called once per window entry; CSS transition animates the change.
  const computeAndSetZoom = useCallback((eventFrame: number) => {
    const event = eventByFrameRef.current.get(eventFrame)
    if (!event) return

    const video = videoRef.current
    if (!video || !video.videoWidth) return

    const { w: containerW, h: containerH } = renderedSizeRef.current
    if (!containerW || !containerH) return

    // Find head tracking data near the impact frame (try ±5 frames)
    let heads: HeadTrack[] = []
    for (let off = 0; off <= 5; off++) {
      heads = trackingRef.current.get(eventFrame + off)
           ?? trackingRef.current.get(eventFrame - off)
           ?? []
      if (heads.length > 0) break
    }

    const involvedHeads = heads.filter(h => event.tracks.includes(h.id))
    if (involvedHeads.length === 0) return  // no tracking data → skip zoom

    const scaleX = containerW / video.videoWidth
    const scaleY = containerH / video.videoHeight

    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity
    for (const h of involvedHeads) {
      const hx  = h.cx * scaleX
      const hy  = h.cy * scaleY
      const pad = h.r * scaleX * 2.5
      minX = Math.min(minX, hx - pad)
      maxX = Math.max(maxX, hx + pad)
      minY = Math.min(minY, hy - pad)
      maxY = Math.max(maxY, hy + pad)
    }

    const boxW = maxX - minX
    const boxH = maxY - minY
    const zoom = Math.min(
      Math.min(containerW / boxW, containerH / boxH) * 0.85,
      4,
    )

    // With transform-origin: 0 0, translate(tx,ty) scale(zoom) maps (x,y) → (tx+zoom*x, ty+zoom*y).
    const hxCenter = (minX + maxX) / 2
    const hyCenter = (minY + maxY) / 2
    const tx = containerW / 2 - zoom * hxCenter
    const ty = containerH / 2 - zoom * hyCenter

    setZoomTransform({ scale: zoom, tx, ty })
  }, [])

  const updatePlayhead = useCallback(() => {
    const v = videoRef.current
    if (v && v.duration) {
      setPlayheadPct((v.currentTime / v.duration) * 100)
      setIsPlaying(!v.paused)
      setCurrentTime(v.currentTime)

      const currentFrame = Math.round(v.currentTime * fps)
      const prevFrame    = prevFrameRef.current
      prevFrameRef.current = currentFrame

      // Coordinate scaling: tracking coords are in original video pixel space
      const canvas = canvasRef.current
      const scaleX = (canvas && v.videoWidth)  ? canvas.width  / v.videoWidth  : 1
      const scaleY = (canvas && v.videoHeight) ? canvas.height / v.videoHeight : 1

      // Impact tracks within ±2 frames for colour coding on head rings
      const impactTracks = new Set<number>()
      for (const ev of eventByFrameRef.current.values()) {
        if (Math.abs(currentFrame - ev.frame) <= 2) {
          for (const tid of ev.tracks) impactTracks.add(tid)
        }
      }

      // Impact detection: fire on every frame crossing (prevFrame → currentFrame),
      // only during active playback (not scrubbing).
      if (!v.paused && currentFrame !== prevFrame) {
        const hit = eventByFrameRef.current.get(currentFrame)
        if (hit) {
          // Always trigger the outer glow flash
          flashStartMsRef.current = Date.now()

          // Pause indefinitely (user must manually resume) only when toggle is on
          if (pauseEnabledRef.current) {
            v.pause()
            setIsFrozen(true)
            freezeEventRef.current   = hit
            freezeStartMsRef.current = Date.now()
          }
        }
      }

      // Outer glow: a subtle red box-shadow that fades over 2.5 s.
      // Applied directly on the container DOM node — no React state, no re-renders.
      const container = videoContainerRef.current
      if (container) {
        const flashAge = flashStartMsRef.current > 0
          ? Date.now() - flashStartMsRef.current
          : Infinity
        const alpha = Math.max(0, 1 - flashAge / 2500)
        if (alpha > 0.005) {
          container.style.boxShadow =
            `0 0 30px 8px rgba(220,38,38,${(alpha * 0.72).toFixed(3)})`
        } else if (container.style.boxShadow) {
          container.style.boxShadow = ''
          flashStartMsRef.current = 0
        }
      }

      // Zoom-on-impact: enter/exit windows
      if (zoomEnabledRef.current) {
        let newWindow: { eventFrame: number } | null = null
        for (const w of impactWindowsRef.current) {
          if (currentFrame >= w.windowStart && currentFrame <= w.windowEnd) {
            newWindow = { eventFrame: w.eventFrame }
            break
          }
        }
        const prev = activeWindowRef.current
        if (newWindow && (!prev || prev.eventFrame !== newWindow.eventFrame)) {
          activeWindowRef.current = newWindow
          computeAndSetZoom(newWindow.eventFrame)
        } else if (!newWindow && prev) {
          activeWindowRef.current = null
          setZoomTransform({ scale: 1, tx: 0, ty: 0 })
        }
      } else if (activeWindowRef.current) {
        activeWindowRef.current = null
        setZoomTransform({ scale: 1, tx: 0, ty: 0 })
      }

      drawOverlay(
        canvas,
        freezeEventRef.current,
        freezeStartMsRef.current,
        profilesRef.current,
        trackingRef.current,
        currentFrame,
        impactTracks,
        scaleX,
        scaleY,
        showImpactHudRef.current,
      )
    }
    rafRef.current = requestAnimationFrame(updatePlayhead)
  }, [fps, computeAndSetZoom])

  useEffect(() => {
    rafRef.current = requestAnimationFrame(updatePlayhead)
    return () => cancelAnimationFrame(rafRef.current)
  }, [updatePlayhead])

  // Sync canvas bitmap dimensions with the video's rendered size
  useEffect(() => {
    const video = videoRef.current
    if (!video) return
    const ro = new ResizeObserver(entries => {
      const { width, height } = entries[0].contentRect
      renderedSizeRef.current = { w: Math.round(width), h: Math.round(height) }
      if (canvasRef.current) {
        canvasRef.current.width  = Math.round(width)
        canvasRef.current.height = Math.round(height)
      }
    })
    ro.observe(video)
    return () => ro.disconnect()
  }, [])

  useImperativeHandle(ref, () => ({
    seekToFrame(frame: number) {
      const v = videoRef.current
      if (v) {
        v.currentTime = frame / fps
        v.pause()
      }
    },
  }))

  function onSeek(frame: number) {
    const v = videoRef.current
    if (v) v.currentTime = frame / fps
  }

  // Resume from a Pause-on-impact freeze. User must call this explicitly.
  function handleResume() {
    videoRef.current?.play().then(() => {
      setIsFrozen(false)
      freezeEventRef.current = null
    }).catch(() => {/* mobile may block; leave isFrozen=true so the hint stays */})
  }

  // Custom control handlers — use onPointerDown + preventDefault to avoid the
  // ghost-click double-toggle that fires when touch simulation is active.
  function handlePlayPausePointerDown(e: React.PointerEvent) {
    e.preventDefault()
    const v = videoRef.current
    if (!v) return
    if (isFrozen) {
      handleResume()
    } else if (v.paused) {
      v.play().catch(() => {})
    } else {
      v.pause()
    }
  }

  function handleFullscreen() {
    const v = videoRef.current
    if (!v) return
    if (document.fullscreenElement) {
      document.exitFullscreen().catch(() => {})
    } else {
      v.requestFullscreen().catch(() => {})
    }
  }

  function formatTime(secs: number) {
    const m = Math.floor(secs / 60)
    const s = Math.floor(secs % 60)
    return `${m}:${s.toString().padStart(2, '0')}`
  }

  function goPrevImpact() {
    const currentFrame = Math.round((videoRef.current?.currentTime ?? 0) * fps)
    const prev = [...eventsSortedRef.current].reverse().find(ev => ev.frame < currentFrame - 5)
    if (prev) onSeek(prev.frame)
  }

  function goNextImpact() {
    const currentFrame = Math.round((videoRef.current?.currentTime ?? 0) * fps)
    const next = eventsSortedRef.current.find(ev => ev.frame > currentFrame + 5)
    if (next) onSeek(next.frame)
  }

  const videoSrc = `/api/video/${jobId}/original`

  const toggleBtnClass = (active: boolean) =>
    `flex items-center gap-1 px-2.5 py-1 sm:px-5 sm:py-3 rounded-md text-sm sm:text-lg font-medium transition-colors
     ${active
       ? 'bg-blue-600 text-white hover:bg-blue-700'
       : 'bg-slate-100 text-slate-700 hover:bg-slate-200'}`

  return (
    <div className="flex flex-col">
      {/* Video + canvas overlay */}
      <div
        ref={videoContainerRef}
        className="relative rounded-xl overflow-hidden bg-black border border-slate-200"
      >
        {/* Zoom wrapper — overflow-hidden on parent clips the zoomed content */}
        <div
          style={{
            transform: `translate(${zoomTransform.tx}px,${zoomTransform.ty}px) scale(${zoomTransform.scale})`,
            transformOrigin: '0 0',
            transition: 'transform 0.6s cubic-bezier(0.4, 0, 0.2, 1)',
            willChange: 'transform',
          }}
        >
          <video
            ref={videoRef}
            src={videoSrc}
            className="w-full block max-h-[33vh] md:max-h-[55vh]"
            playsInline
            onLoadedMetadata={() => setDuration(videoRef.current?.duration ?? 0)}
            onPointerDown={handlePlayPausePointerDown}
            style={{ touchAction: 'manipulation', cursor: 'pointer' }}
          />
          <canvas
            ref={canvasRef}
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: '100%',
              pointerEvents: 'none',
            }}
          />
        </div>

        {/* Custom controls bar — outside zoom wrapper so it stays in place */}
        <div className="absolute bottom-0 left-0 right-0 flex items-center gap-3 px-3 py-2 bg-gradient-to-t from-black/70 to-transparent">
          <button
            onPointerDown={handlePlayPausePointerDown}
            className="text-white p-1 rounded focus:outline-none"
            style={{ touchAction: 'manipulation' }}
            aria-label={isPlaying ? 'Pause' : 'Play'}
          >
            {isPlaying && !isFrozen
              ? <Pause className="w-5 h-5 sm:w-6 sm:h-6 fill-white" />
              : <Play  className="w-5 h-5 sm:w-6 sm:h-6 fill-white" />
            }
          </button>

          <span className="text-white text-xs sm:text-base tabular-nums select-none">
            {formatTime(currentTime)}{duration > 0 ? ` / ${formatTime(duration)}` : ''}
          </span>

          <div className="flex-1" />

          <button
            onPointerDown={e => { e.preventDefault(); handleFullscreen() }}
            className="text-white p-1 rounded focus:outline-none"
            style={{ touchAction: 'manipulation' }}
            aria-label="Fullscreen"
          >
            <Maximize2 className="w-4 h-4 sm:w-5 sm:h-5" />
          </button>
        </div>

        {/* Tap-to-resume hint — shown when Pause toggle fired */}
        {isFrozen && (
          <div className="absolute inset-0 flex items-end justify-center pb-3 pointer-events-none">
            <div className="flex items-center gap-2 bg-black/60 backdrop-blur-sm rounded-full px-4 py-2">
              <Play className="w-4 h-4 text-white fill-white" />
              <span className="text-white text-sm font-medium">Tap to continue</span>
            </div>
          </div>
        )}
      </div>

      {/* Timeline */}
      <ImpactTimeline
        events={events}
        profiles={profiles}
        totalFrames={totalFrames}
        fps={fps}
        playheadPct={playheadPct}
        videoSrc={videoSrc}
        onSeek={onSeek}
        onMarkerClick={onMarkerClick}
      />

      {/* Impact controls */}
      <div className="flex items-center gap-3 mt-2 px-1">

        {/* go to: box */}
        <div className="border border-slate-200 rounded-lg px-2 pt-1 pb-1.5 sm:px-3 sm:pt-2 sm:pb-2.5">
          <div className="text-[10px] sm:text-sm text-slate-400 uppercase tracking-wide font-medium mb-1">go to:</div>
          <div className="flex gap-1">
            <button
              onPointerDown={e => { e.preventDefault(); goPrevImpact() }}
              disabled={events.length === 0}
              className="flex items-center gap-1 px-2.5 py-1 sm:px-5 sm:py-3 rounded-md text-sm sm:text-lg font-medium
                         bg-slate-100 hover:bg-slate-200 text-slate-700 disabled:opacity-40 transition-colors"
              style={{ touchAction: 'manipulation' }}
              aria-label="Previous impact"
            >
              <ChevronLeft className="w-3.5 h-3.5 sm:w-5 sm:h-5" />
              Prev
            </button>
            <button
              onPointerDown={e => { e.preventDefault(); goNextImpact() }}
              disabled={events.length === 0}
              className="flex items-center gap-1 px-2.5 py-1 sm:px-5 sm:py-3 rounded-md text-sm sm:text-lg font-medium
                         bg-slate-100 hover:bg-slate-200 text-slate-700 disabled:opacity-40 transition-colors"
              style={{ touchAction: 'manipulation' }}
              aria-label="Next impact"
            >
              Next
              <ChevronRight className="w-3.5 h-3.5 sm:w-5 sm:h-5" />
            </button>
          </div>
        </div>

        <div className="flex-1" />

        {/* on impact: box */}
        <div className="border border-slate-200 rounded-lg px-2 pt-1 pb-1.5 sm:px-3 sm:pt-2 sm:pb-2.5">
          <div className="text-[10px] sm:text-sm text-slate-400 uppercase tracking-wide font-medium mb-1">on impact:</div>
          <div className="flex gap-1">
            <button
              onPointerDown={e => { e.preventDefault(); setZoomEnabled(v => !v) }}
              className={toggleBtnClass(zoomEnabled)}
              style={{ touchAction: 'manipulation' }}
              aria-label="Toggle zoom on impact"
              aria-pressed={zoomEnabled}
            >
              <ZoomIn className="w-3.5 h-3.5 sm:w-5 sm:h-5" />
              Zoom
            </button>
            <button
              onPointerDown={e => { e.preventDefault(); setPauseEnabled(v => !v) }}
              className={toggleBtnClass(pauseEnabled)}
              style={{ touchAction: 'manipulation' }}
              aria-label="Toggle pause on impact"
              aria-pressed={pauseEnabled}
            >
              <PauseCircle className="w-3.5 h-3.5 sm:w-5 sm:h-5" />
              Pause
            </button>
          </div>
        </div>

      </div>
    </div>
  )
})

export default VideoPlayer
