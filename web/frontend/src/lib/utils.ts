import { type ClassValue, clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'
import type { RiskLevel } from '@/types/impact'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function frameToTimecode(frame: number, fps: number): string {
  const totalSecs = frame / fps
  const mins = Math.floor(totalSecs / 60).toString().padStart(2, '0')
  const secs = Math.floor(totalSecs % 60).toString().padStart(2, '0')
  return `${mins}:${secs}`
}

export function riskBg(risk: RiskLevel): string {
  switch (risk) {
    case 'HIGH':     return 'bg-red-600'
    case 'ELEVATED': return 'bg-amber-500'
    case 'LOW':      return 'bg-green-600'
  }
}

export function riskText(risk: RiskLevel): string {
  switch (risk) {
    case 'HIGH':     return 'text-red-400'
    case 'ELEVATED': return 'text-amber-400'
    case 'LOW':      return 'text-green-400'
  }
}

export function riskBorder(risk: RiskLevel): string {
  switch (risk) {
    case 'HIGH':     return 'border-red-600'
    case 'ELEVATED': return 'border-amber-500'
    case 'LOW':      return 'border-green-600'
  }
}

export function markerColor(risk: RiskLevel): string {
  switch (risk) {
    case 'HIGH':     return '#ef4444'
    case 'ELEVATED': return '#f59e0b'
    case 'LOW':      return '#22c55e'
  }
}

const RISK_ORDER: Record<RiskLevel, number> = { LOW: 0, ELEVATED: 1, HIGH: 2 }

export function worstRisk(risks: RiskLevel[]): RiskLevel {
  if (risks.length === 0) return 'LOW'
  return risks.reduce((a, b) => (RISK_ORDER[b] > RISK_ORDER[a] ? b : a))
}
