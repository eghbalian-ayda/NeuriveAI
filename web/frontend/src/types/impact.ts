export type RiskLevel = 'LOW' | 'ELEVATED' | 'HIGH'

export interface ImpactEvent {
  frame: number
  tracks: number[]
  confidence: number
  stages: string[]
  details: Record<string, number>
}

export interface ImpactProfile {
  track_id: number
  event_frame: number
  n_frames: number
  frame_indices: number[]
  omega_magnitudes: number[]
  omega_peak_rad_s: number
  alpha_peak_rad_s2: number
  bric_r: number
  bric_r_risk: RiskLevel
  klc_rot_rad_s: number
  klc_risk: RiskLevel
  damage: number
  damage_risk: RiskLevel
  risk_summary: RiskLevel
  delta_omega_rad_s: number
  pulse_duration_s: number
}

export interface ImpactReport {
  events: ImpactEvent[]
  profiles: ImpactProfile[]
}

export interface ResultsData {
  fps: number
  total_frames: number
  report: ImpactReport
}

export interface PlayerCard {
  trackId: number
  impactCount: number
  highestRisk: RiskLevel
  impacts: Array<{ profile: ImpactProfile; event: ImpactEvent }>
}
