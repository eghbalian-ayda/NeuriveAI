import { useState, useEffect } from 'react'
import { Users, Download, ChevronRight } from 'lucide-react'
import type { PlayerCard, ResultsData } from '@/types/impact'
import { riskBg, riskText } from '@/lib/utils'
import PlayerDetailModal from './PlayerDetailModal'

interface Props {
  players: PlayerCard[]
  fps: number
  highlightedTracks: number[]
  data: ResultsData
  onPlayerSeek: (frame: number) => void
}

export default function PlayerRoster({ players, fps, highlightedTracks, data, onPlayerSeek }: Props) {
  const [openPlayerId, setOpenPlayerId] = useState<number | null>(null)
  const [ringTracks, setRingTracks] = useState<Set<number>>(new Set())

  // Apply 2s highlight ring when highlighted tracks change
  useEffect(() => {
    if (highlightedTracks.length === 0) return
    setRingTracks(new Set(highlightedTracks))
    const timer = setTimeout(() => setRingTracks(new Set()), 2000)
    return () => clearTimeout(timer)
  }, [highlightedTracks])

  function exportJson() {
    const blob = new Blob([JSON.stringify(data.report, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'neurive_impact_report.json'
    a.click()
    URL.revokeObjectURL(url)
  }

  const openPlayer = players.find(p => p.trackId === openPlayerId) ?? null

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center gap-2 mb-4">
        <Users className="w-4 h-4 text-slate-400" />
        <h3 className="text-slate-200 font-semibold text-sm">Players Involved</h3>
        <span className="ml-auto text-slate-600 text-xs">{players.length} player{players.length !== 1 ? 's' : ''}</span>
      </div>

      {/* Player cards */}
      <div className="flex flex-col gap-2 flex-1 overflow-y-auto scrollbar-thin pr-1">
        {players.length === 0 ? (
          <p className="text-slate-600 text-sm text-center mt-8">No impacts detected</p>
        ) : (
          players.map((player) => {
            const hasRing = ringTracks.has(player.trackId)
            return (
              <button
                key={player.trackId}
                onClick={() => setOpenPlayerId(player.trackId)}
                className={`
                  w-full text-left bg-slate-900 rounded-xl p-4 border transition-all duration-150
                  hover:bg-slate-800 hover:border-slate-600 group
                  ${hasRing
                    ? 'border-blue-500 shadow-[0_0_0_3px_rgba(59,130,246,0.25)]'
                    : 'border-slate-800'}
                `}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-2xl font-bold text-white">#{player.trackId}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className={`
                      px-2 py-0.5 rounded-full text-xs font-bold
                      ${riskBg(player.highestRisk)} text-white
                    `}>
                      {player.highestRisk}
                    </span>
                    <ChevronRight className="w-4 h-4 text-slate-600 group-hover:text-slate-400 transition-colors" />
                  </div>
                </div>
                <p className={`text-xs mt-1 ${riskText(player.highestRisk)}`}>
                  {player.impactCount} impact{player.impactCount !== 1 ? 's' : ''} detected
                </p>
              </button>
            )
          })
        )}
      </div>

      {/* Export */}
      <div className="mt-4 pt-4 border-t border-slate-800">
        <button
          onClick={exportJson}
          className="
            w-full flex items-center justify-center gap-2 py-2.5 rounded-xl
            border border-slate-700 text-slate-400 text-sm
            hover:border-slate-500 hover:text-slate-200 transition-colors
          "
        >
          <Download className="w-4 h-4" />
          Export Report JSON
        </button>
      </div>

      {/* Modal */}
      {openPlayer && (
        <PlayerDetailModal
          player={openPlayer}
          fps={fps}
          events={data.report.events}
          onClose={() => setOpenPlayerId(null)}
          onSeek={(frame) => {
            setOpenPlayerId(null)
            onPlayerSeek(frame)
          }}
        />
      )}
    </div>
  )
}
