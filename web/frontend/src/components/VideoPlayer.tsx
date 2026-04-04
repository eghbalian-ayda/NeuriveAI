import { useEffect, useRef, useState, useCallback, forwardRef, useImperativeHandle } from 'react'
import type { ImpactEvent, ImpactProfile } from '@/types/impact'
import ImpactTimeline from './ImpactTimeline'

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
}

type VideoMode = 'annotated' | 'original'

const VideoPlayer = forwardRef<VideoPlayerHandle, Props>(function VideoPlayer(
  { jobId, events, profiles, totalFrames, fps, onMarkerClick },
  ref,
) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [mode, setMode] = useState<VideoMode>('annotated')
  const [playheadPct, setPlayheadPct] = useState(0)
  const rafRef = useRef<number>(0)

  function switchMode(newMode: VideoMode) {
    const v = videoRef.current
    const savedTime = v?.currentTime ?? 0
    setMode(newMode)
    // Restore time after metadata loads
    const handler = () => {
      if (v) {
        v.currentTime = savedTime
        v.removeEventListener('loadedmetadata', handler)
      }
    }
    v?.addEventListener('loadedmetadata', handler)
  }

  const updatePlayhead = useCallback(() => {
    const v = videoRef.current
    if (v && v.duration) {
      setPlayheadPct((v.currentTime / v.duration) * 100)
    }
    rafRef.current = requestAnimationFrame(updatePlayhead)
  }, [])

  useEffect(() => {
    rafRef.current = requestAnimationFrame(updatePlayhead)
    return () => cancelAnimationFrame(rafRef.current)
  }, [updatePlayhead])

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
    if (v) {
      v.currentTime = frame / fps
    }
  }

  const videoSrc = `/api/video/${jobId}/${mode}`

  return (
    <div className="flex flex-col">
      {/* Mode toggle */}
      <div className="flex items-center justify-between mb-3">
        <span className="text-slate-400 text-sm font-medium">Video</span>
        <div className="flex bg-slate-900 rounded-lg p-1 border border-slate-800">
          {(['annotated', 'original'] as VideoMode[]).map((m) => (
            <button
              key={m}
              onClick={() => switchMode(m)}
              className={`
                px-3 py-1 rounded-md text-xs font-medium transition-colors capitalize
                ${mode === m
                  ? 'bg-slate-700 text-white'
                  : 'text-slate-500 hover:text-slate-300'}
              `}
            >
              {m}
            </button>
          ))}
        </div>
      </div>

      {/* Video */}
      <div className="relative rounded-xl overflow-hidden bg-black border border-slate-800">
        <video
          ref={videoRef}
          src={videoSrc}
          controls
          className="w-full block"
          style={{ maxHeight: '55vh' }}
          playsInline
        />
      </div>

      {/* Timeline */}
      <ImpactTimeline
        events={events}
        profiles={profiles}
        totalFrames={totalFrames}
        fps={fps}
        playheadPct={playheadPct}
        onSeek={onSeek}
        onMarkerClick={onMarkerClick}
      />
    </div>
  )
})

export default VideoPlayer
