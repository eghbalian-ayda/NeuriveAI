import { useRef, useState, useMemo } from 'react'
import { RotateCcw, AlertTriangle, TrendingUp, Target } from 'lucide-react'
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
    <div className="bg-white border border-slate-200 rounded-xl px-4 py-3 flex items-center gap-3">
      <Icon className="w-5 h-5 text-slate-400 shrink-0" />
      <div>
        <p className={`text-xl font-bold ${valueClass ?? 'text-slate-900'}`}>{value}</p>
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
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {/* Top bar */}
      <header className="sticky top-0 z-30 bg-white/95 backdrop-blur border-b border-slate-200">
        <div className="max-w-screen-2xl mx-auto px-4 sm:px-6 py-2 sm:py-3 flex items-center gap-4">
          {/* Logo */}
          <div className="flex items-center gap-2 mr-4">
            <img src="/logo.png" alt="NeuriveAI logo" className="w-7 h-7 sm:w-10 sm:h-10 object-contain" />
            <span className="text-slate-900 font-bold text-base sm:text-2xl">NeuriveAI</span>
          </div>

          {/* Stats — desktop only */}
          <div className="hidden md:flex gap-3 flex-1 overflow-x-auto">
            <StatCard
              icon={AlertTriangle}
              label="Total Impacts"
              value={totalImpacts}
              valueClass={totalImpacts > 0 ? 'text-red-500' : 'text-slate-900'}
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

          {/* Spacer on mobile */}
          <div className="flex-1 md:hidden" />

          {/* Reset */}
          <button
            onClick={onReset}
            className="
              flex items-center gap-2 px-3 py-1.5 sm:px-6 sm:py-3 rounded-xl border border-slate-300
              text-slate-500 text-sm sm:text-xl hover:border-slate-400 hover:text-slate-700
              transition-colors shrink-0 bg-white
            "
          >
            <RotateCcw className="w-4 h-4 sm:w-6 sm:h-6" />
            New Analysis
          </button>
        </div>
      </header>

      {/* Main content */}
      <div className="flex-1 max-w-screen-2xl mx-auto w-full px-4 md:px-6 py-4 md:py-6 flex flex-col md:grid md:grid-cols-[1fr_340px] gap-4 md:gap-6">
        {/* Left: video + timeline + mobile stats */}
        <div className="flex flex-col gap-4">
          <VideoPlayer
            ref={videoRef}
            jobId={jobId}
            events={report.events}
            profiles={report.profiles}
            totalFrames={total_frames}
            fps={fps}
            onMarkerClick={handleMarkerClick}
            tracking={report.tracking}
          />

          {/* Compact stats — mobile only */}
          <div className="flex md:hidden bg-white border border-slate-200 rounded-xl">
            <div className="flex-1 flex items-center gap-3 px-4 py-3">
              <AlertTriangle className="w-5 h-5 sm:w-7 sm:h-7 text-slate-400 shrink-0" />
              <div>
                <p className={`text-xl sm:text-3xl font-bold ${totalImpacts > 0 ? 'text-red-500' : 'text-slate-900'}`}>{totalImpacts}</p>
                <p className="text-slate-500 text-xs sm:text-base">Total Impacts</p>
              </div>
            </div>
            {highestRiskPlayer && (
              <div className="flex-1 flex items-center gap-3 px-4 py-3 border-l border-slate-200">
                <Target className="w-5 h-5 sm:w-7 sm:h-7 text-slate-400 shrink-0" />
                <div>
                  <p className={`text-xl sm:text-3xl font-bold ${riskText(highestRiskPlayer.highestRisk)}`}>#{highestRiskPlayer.trackId}</p>
                  <p className="text-slate-500 text-xs sm:text-base">Highest Risk</p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right: player roster */}
        <div className="bg-white border border-slate-200 rounded-2xl p-4 sm:p-6 flex flex-col md:max-h-[calc(100vh-100px)] md:min-h-[400px]">
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
