import * as Dialog from '@radix-ui/react-dialog'
import { X, Play } from 'lucide-react'
import {
  AreaChart, Area, XAxis, YAxis, Tooltip as ReTooltip,
  ResponsiveContainer, ReferenceLine,
} from 'recharts'
import type { ImpactEvent, PlayerCard } from '@/types/impact'
import { frameToTimecode, riskBg, riskText } from '@/lib/utils'

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
  const textClass = risk === 'HIGH' ? 'text-red-400' : risk === 'ELEVATED' ? 'text-amber-400' : 'text-white'
  return (
    <div className="bg-slate-950 rounded-lg p-3 text-center">
      <p className={`text-base font-bold font-mono ${textClass}`}>
        {value.toFixed(value < 10 ? 4 : 1)}{unit}
      </p>
      <p className="text-slate-500 text-[10px] mt-0.5">{label}</p>
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
        <Dialog.Overlay className="fixed inset-0 bg-black/60 backdrop-blur-sm z-40 animate-fade-in" />
        <Dialog.Content
          className="
            fixed inset-y-4 right-4 left-4 md:left-auto md:w-[520px]
            bg-slate-900 border border-slate-700 rounded-2xl z-50
            flex flex-col shadow-2xl animate-fade-in overflow-hidden
          "
        >
          {/* Header */}
          <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800">
            <div>
              <Dialog.Title className="text-white font-bold text-lg">
                Player #{player.trackId}
              </Dialog.Title>
              <Dialog.Description className="text-slate-400 text-sm">
                {player.impactCount} impact{player.impactCount !== 1 ? 's' : ''} · Highest risk:{' '}
                <span className={riskText(player.highestRisk)}>{player.highestRisk}</span>
              </Dialog.Description>
            </div>
            <Dialog.Close asChild>
              <button className="text-slate-500 hover:text-slate-200 transition-colors p-1 rounded-lg hover:bg-slate-800">
                <X className="w-5 h-5" />
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
                  className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden"
                >
                  {/* Impact header */}
                  <div className="flex items-center justify-between px-4 py-3 border-b border-slate-700">
                    <div className="flex items-center gap-3">
                      <span className="text-slate-200 font-mono text-sm font-semibold">{timecode}</span>
                      <span className={`
                        px-2 py-0.5 rounded-full text-xs font-bold
                        ${riskBg(profile.risk_summary)} text-white
                      `}>
                        {profile.risk_summary}
                      </span>
                    </div>
                    <button
                      onClick={() => onSeek(profile.event_frame)}
                      className="
                        flex items-center gap-1.5 px-3 py-1.5 rounded-lg
                        bg-blue-600 hover:bg-blue-700 text-white text-xs font-medium
                        transition-colors
                      "
                    >
                      <Play className="w-3 h-3" />
                      Seek to {timecode}
                    </button>
                  </div>

                  <div className="p-4 space-y-4">
                    {/* Metrics grid */}
                    <div className="grid grid-cols-3 gap-2">
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
                      <div className="flex flex-wrap gap-1.5">
                        {event.stages.map(stage => (
                          <span
                            key={stage}
                            className="
                              px-2 py-0.5 bg-blue-950 border border-blue-800
                              text-blue-300 rounded-full text-[10px] font-mono uppercase
                            "
                          >
                            {stage.replace('_', ' ')}
                          </span>
                        ))}
                        <span className="px-2 py-0.5 bg-slate-700 text-slate-400 rounded-full text-[10px]">
                          conf: {Math.round(event.confidence * 100)}%
                        </span>
                      </div>
                    )}

                    {/* Sparkline */}
                    {chartData.length > 1 && (
                      <div>
                        <p className="text-slate-500 text-[10px] mb-1.5 uppercase tracking-wide">
                          Angular velocity ‖ω‖ over impact window
                        </p>
                        <div className="h-20">
                          <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={chartData} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
                              <defs>
                                <linearGradient id={`omegaGrad-${i}`} x1="0" y1="0" x2="0" y2="1">
                                  <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                                  <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                                </linearGradient>
                              </defs>
                              <XAxis dataKey="t" hide />
                              <YAxis
                                tick={{ fontSize: 9, fill: '#64748b' }}
                                width={28}
                                tickLine={false}
                                axisLine={false}
                              />
                              <ReTooltip
                                contentStyle={{
                                  background: '#1e293b',
                                  border: '1px solid #334155',
                                  borderRadius: '8px',
                                  fontSize: '11px',
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
                        <p className="text-slate-600 text-[9px] text-right mt-0.5">
                          red dashed = peak ({peakOmega.toFixed(2)} rad/s)
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  )
}
