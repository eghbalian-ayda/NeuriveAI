import { useRef, useState, useMemo } from 'react'
import { Zap, RotateCcw, AlertTriangle, TrendingUp, Target } from 'lucide-react'
import type { ResultsData, PlayerCard } from '@/types/impact'
import { worstRisk, riskText } from '@/lib/utils'
import VideoPlayer, { type VideoPlayerHandle } from './VideoPlayer'
import PlayerRoster from './PlayerRoster'

interface Props {
  jobId: string
  data: ResultsData
  onReset: () => void
}

function StatCard({ icon: Icon, label, value, valueClass }: {
  icon: React.ElementType
  label: string
  value: string | number
  valueClass?: string
}) {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl px-4 py-3 flex items-center gap-3">
      <Icon className="w-5 h-5 text-slate-500 shrink-0" />
      <div>
        <p className={`text-xl font-bold ${valueClass ?? 'text-white'}`}>{value}</p>
        <p className="text-slate-500 text-xs">{label}</p>
      </div>
    </div>
  )
}

export default function ResultsScreen({ jobId, data, onReset }: Props) {
  const { fps, total_frames, report } = data
  const videoRef = useRef<VideoPlayerHandle>(null)
  const [highlightedTracks, setHighlightedTracks] = useState<number[]>([])

  // Build player cards
  const playerCards = useMemo<PlayerCard[]>(() => {
    const byPlayer: Record<number, PlayerCard> = {}
    for (const profile of report.profiles) {
      if (!byPlayer[profile.track_id]) {
        byPlayer[profile.track_id] = {
          trackId: profile.track_id,
          impactCount: 0,
          highestRisk: 'LOW',
          impacts: [],
        }
      }
      const ev = report.events.find(
        e => e.frame === profile.event_frame && e.tracks.includes(profile.track_id)
      ) ?? null
      byPlayer[profile.track_id].impacts.push({ profile, event: ev! })
      byPlayer[profile.track_id].impactCount++
    }
    return Object.values(byPlayer).map(p => ({
      ...p,
      highestRisk: worstRisk(p.impacts.map(i => i.profile.risk_summary)),
    })).sort((a, b) => {
      const riskOrder = { HIGH: 2, ELEVATED: 1, LOW: 0 }
      return riskOrder[b.highestRisk] - riskOrder[a.highestRisk]
    })
  }, [report])

  // Summary stats
  const totalImpacts = report.events.length
  const highestRiskPlayer = playerCards[0]
  const avgConf = report.events.length > 0
    ? Math.round(report.events.reduce((s, e) => s + e.confidence, 0) / report.events.length * 100)
    : 0

  function handleMarkerClick(trackIds: number[]) {
    setHighlightedTracks(trackIds)
    if (trackIds.length > 0) {
      setTimeout(() => setHighlightedTracks([]), 2100)
    }
  }

  function handlePlayerSeek(frame: number) {
    videoRef.current?.seekToFrame(frame)
  }

  return (
    <div className="min-h-screen bg-slate-950 flex flex-col">
      {/* Top bar */}
      <header className="sticky top-0 z-30 bg-slate-950/95 backdrop-blur border-b border-slate-800">
        <div className="max-w-screen-2xl mx-auto px-6 py-3 flex items-center gap-4">
          {/* Logo */}
          <div className="flex items-center gap-2 mr-4">
            <div className="w-7 h-7 rounded-lg bg-blue-600 flex items-center justify-center">
              <Zap className="w-4 h-4 text-white" />
            </div>
            <span className="text-white font-bold text-base">NeuriveAI</span>
          </div>

          {/* Stats */}
          <div className="flex gap-3 flex-1 overflow-x-auto">
            <StatCard
              icon={AlertTriangle}
              label="Total Impacts"
              value={totalImpacts}
              valueClass={totalImpacts > 0 ? 'text-red-400' : 'text-white'}
            />
            {highestRiskPlayer && (
              <StatCard
                icon={Target}
                label="Highest Risk Player"
                value={`#${highestRiskPlayer.trackId}`}
                valueClass={riskText(highestRiskPlayer.highestRisk)}
              />
            )}
            <StatCard
              icon={TrendingUp}
              label="Avg Detection Confidence"
              value={`${avgConf}%`}
            />
          </div>

          {/* Reset */}
          <button
            onClick={onReset}
            className="
              flex items-center gap-2 px-4 py-2 rounded-xl border border-slate-700
              text-slate-400 text-sm hover:border-slate-500 hover:text-slate-200
              transition-colors shrink-0
            "
          >
            <RotateCcw className="w-4 h-4" />
            New Analysis
          </button>
        </div>
      </header>

      {/* Main content */}
      <div className="flex-1 max-w-screen-2xl mx-auto w-full px-6 py-6 grid gap-6"
        style={{ gridTemplateColumns: '1fr 340px' }}
      >
        {/* Left: video + timeline */}
        <VideoPlayer
          ref={videoRef}
          jobId={jobId}
          events={report.events}
          profiles={report.profiles}
          totalFrames={total_frames}
          fps={fps}
          onMarkerClick={handleMarkerClick}
        />

        {/* Right: player roster */}
        <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-4 flex flex-col"
          style={{ maxHeight: 'calc(100vh - 100px)', minHeight: '400px' }}
        >
          <PlayerRoster
            players={playerCards}
            fps={fps}
            highlightedTracks={highlightedTracks}
            data={data}
            onPlayerSeek={handlePlayerSeek}
          />
        </div>
      </div>
    </div>
  )
}
