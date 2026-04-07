import { lazy, Suspense } from 'react'
import * as Dialog from '@radix-ui/react-dialog'
import { X, Play } from 'lucide-react'
import {
  AreaChart, Area, XAxis, YAxis, Tooltip as ReTooltip,
  ResponsiveContainer, ReferenceLine,
} from 'recharts'
import type { ImpactEvent, PlayerCard } from '@/types/impact'
import { frameToTimecode, riskBg, riskText } from '@/lib/utils'

const BrainVisualization = lazy(() => import('./BrainVisualization'))

interface Props {
  player: PlayerCard
  fps: number
  events: ImpactEvent[]
  onClose: () => void
  onSeek: (frame: number) => void
}


function MetricCell({ label, value, unit, risk }: {
  label: string; value: number; unit: string; risk?: string
}) {
  const textClass = risk === 'HIGH' ? 'text-red-500' : risk === 'ELEVATED' ? 'text-amber-500' : 'text-slate-900'
  return (
    <div className="bg-gray-50 rounded-lg p-3 text-center">
      <p className={`text-base sm:text-2xl font-bold font-mono ${textClass}`}>
        {value.toFixed(value < 10 ? 4 : 1)}{unit}
      </p>
      <p className="text-slate-400 text-[10px] sm:text-base mt-0.5">{label}</p>
    </div>
  )
}

export default function PlayerDetailModal({ player, fps, events, onClose, onSeek }: Props) {
  const sortedImpacts = [...player.impacts].sort(
    (a, b) => a.profile.event_frame - b.profile.event_frame
  )

  return (
    <Dialog.Root open onOpenChange={(open) => { if (!open) onClose() }}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 bg-black/40 backdrop-blur-sm z-40 animate-fade-in" />
        <Dialog.Content
          className="
            fixed inset-0 md:inset-y-4 md:right-4 md:left-auto md:w-[520px]
            bg-white border border-slate-200 rounded-none md:rounded-2xl z-50
            flex flex-col shadow-2xl animate-fade-in overflow-hidden
          "
        >
          {/* Header */}
          <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200">
            <div>
              <Dialog.Title className="text-slate-900 font-bold text-lg sm:text-2xl">
                Player #{player.trackId}
              </Dialog.Title>
              <Dialog.Description className="text-slate-500 text-sm sm:text-lg">
                {player.impactCount} impact{player.impactCount !== 1 ? 's' : ''} · Highest risk:{' '}
                <span className={riskText(player.highestRisk)}>{player.highestRisk}</span>
              </Dialog.Description>
            </div>
            <Dialog.Close asChild>
              <button className="text-slate-400 hover:text-slate-700 transition-colors p-1 rounded-lg hover:bg-slate-100">
                <X className="w-5 h-5 sm:w-7 sm:h-7" />
              </button>
            </Dialog.Close>
          </div>

          {/* Scrollable body */}
          <div className="flex-1 overflow-y-auto scrollbar-thin px-6 py-4 space-y-4">
            {sortedImpacts.map(({ profile, event }, i) => {
              const timecode = frameToTimecode(profile.event_frame, fps)
              const chartData = profile.frame_indices.slice(0, profile.omega_magnitudes.length).map(
                (f, idx) => ({ t: f, omega: profile.omega_magnitudes[idx] })
              )
              const peakOmega = profile.omega_peak_rad_s

              return (
                <div
                  key={i}
                  className="bg-slate-50 rounded-xl border border-slate-200 overflow-hidden"
                >
                  {/* Impact header */}
                  <div className="flex items-center justify-between px-4 py-3 sm:px-5 sm:py-4 border-b border-slate-200">
                    <div className="flex items-center gap-3">
                      <span className="text-slate-700 font-mono text-sm sm:text-lg font-semibold">{timecode}</span>
                      <span className={`
                        px-2 py-0.5 sm:px-3 sm:py-1 rounded-full text-xs sm:text-sm font-bold
                        ${riskBg(profile.risk_summary)} text-white
                      `}>
                        {profile.risk_summary}
                      </span>
                    </div>
                    <button
                      onClick={() => onSeek(profile.event_frame)}
                      className="
                        flex items-center gap-1.5 px-3 py-1.5 sm:px-5 sm:py-3 rounded-lg
                        bg-blue-600 hover:bg-blue-700 text-white text-xs sm:text-base font-medium
                        transition-colors
                      "
                    >
                      <Play className="w-3 h-3 sm:w-4 sm:h-4" />
                      Seek to {timecode}
                    </button>
                  </div>

                  <div className="p-4 sm:p-5 space-y-4">
                    {/* Metrics grid */}
                    <div className="grid grid-cols-3 gap-2 sm:gap-3">
                      <MetricCell
                        label="ω peak (rad/s)"
                        value={profile.omega_peak_rad_s}
                        unit=""
                      />
                      <MetricCell
                        label="α peak (rad/s²)"
                        value={profile.alpha_peak_rad_s2}
                        unit=""
                      />
                      <MetricCell
                        label="Pulse duration"
                        value={profile.pulse_duration_s * 1000}
                        unit=" ms"
                      />
                      <MetricCell
                        label="BrIC_R"
                        value={profile.bric_r}
                        unit=""
                        risk={profile.bric_r_risk}
                      />
                      <MetricCell
                        label="KLC (rad/s)"
                        value={profile.klc_rot_rad_s}
                        unit=""
                        risk={profile.klc_risk}
                      />
                      <MetricCell
                        label="DAMAGE"
                        value={profile.damage}
                        unit=""
                        risk={profile.damage_risk}
                      />
                    </div>

                    {/* Detection stage pills */}
                    {event && (
                      <div className="flex flex-wrap gap-1.5 sm:gap-2">
                        {event.stages.map(stage => (
                          <span
                            key={stage}
                            className="
                              px-2 py-0.5 sm:px-3 sm:py-1 bg-blue-50 border border-blue-200
                              text-blue-600 rounded-full text-[10px] sm:text-sm font-mono uppercase
                            "
                          >
                            {stage.replace('_', ' ')}
                          </span>
                        ))}
                        <span className="px-2 py-0.5 sm:px-3 sm:py-1 bg-slate-100 text-slate-500 rounded-full text-[10px] sm:text-sm">
                          conf: {Math.round(event.confidence * 100)}%
                        </span>
                      </div>
                    )}

                    {/* Sparkline */}
                    {chartData.length > 1 && (
                      <div>
                        <p className="text-slate-400 text-[10px] sm:text-sm mb-1.5 uppercase tracking-wide">
                          Angular velocity ‖ω‖ over impact window
                        </p>
                        <div className="h-20 sm:h-32">
                          <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={chartData} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
                              <defs>
                                <linearGradient id={`omegaGrad-${i}`} x1="0" y1="0" x2="0" y2="1">
                                  <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.2} />
                                  <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                                </linearGradient>
                              </defs>
                              <XAxis dataKey="t" hide />
                              <YAxis
                                tick={{ fontSize: 11, fill: '#94a3b8' }}
                                width={32}
                                tickLine={false}
                                axisLine={false}
                              />
                              <ReTooltip
                                contentStyle={{
                                  background: '#ffffff',
                                  border: '1px solid #e2e8f0',
                                  borderRadius: '8px',
                                  fontSize: '13px',
                                  padding: '4px 8px',
                                }}
                                formatter={(v: number) => [`${v.toFixed(3)} rad/s`, '‖ω‖']}
                                labelFormatter={(t: number) => `Frame ${t}`}
                              />
                              <ReferenceLine
                                y={peakOmega}
                                stroke="#ef4444"
                                strokeDasharray="3 3"
                                strokeWidth={1}
                              />
                              <Area
                                type="monotone"
                                dataKey="omega"
                                stroke="#3b82f6"
                                strokeWidth={1.5}
                                fill={`url(#omegaGrad-${i})`}
                                dot={false}
                                activeDot={{ r: 3, fill: '#3b82f6' }}
                              />
                            </AreaChart>
                          </ResponsiveContainer>
                        </div>
                        <p className="text-slate-400 text-[9px] sm:text-xs text-right mt-0.5">
                          red dashed = peak ({peakOmega.toFixed(2)} rad/s)
                        </p>
                      </div>
                    )}
                  </div>

                  {/* 3D Brain Impact Map */}
                  {profile.regional_tbi_probs && (
                    <div className="border-t border-slate-200 overflow-hidden">
                      <div className="flex items-center justify-between px-4 py-2 sm:px-5 sm:py-3 bg-slate-100">
                        <span className="text-[10px] sm:text-sm text-slate-500 font-medium uppercase tracking-wide">
                          Brain Region Impact Map
                        </span>
                        {profile.tbi_probability_pct !== undefined && (
                          <span className="text-[10px] sm:text-sm font-mono font-bold text-slate-600">
                            {profile.tbi_probability_pct.toFixed(1)}% TBI probability
                          </span>
                        )}
                      </div>
                      <Suspense fallback={
                        <div className="aspect-square w-[80%] max-w-[375px] mx-auto bg-slate-50 flex items-center justify-center">
                          <div className="w-10 h-10 rounded-full border border-slate-300 border-t-blue-400 animate-spin" />
                        </div>
                      }>
                        <BrainVisualization
                          regionalProbs={profile.regional_tbi_probs}
                          omegaMagnitudes={profile.omega_magnitudes}
                          omegaUnitVectors={profile.omega_unit_vectors}
                        />
                      </Suspense>
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  )
}
