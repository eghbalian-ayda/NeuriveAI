import { useEffect, useRef, useState } from 'react'
import { Zap, X } from 'lucide-react'
import type { ResultsData } from '@/types/impact'

type Stage = 'uploading' | 'pass1' | 'pass2' | 'rendering' | 'complete' | 'error'

const STAGE_LABELS: Record<Stage, string> = {
  uploading:  'Upload',
  pass1:      'Pass 1 — Frame Scan',
  pass2:      'Pass 2 — Brain Analysis',
  rendering:  'Rendering',
  complete:   'Complete',
  error:      'Error',
}

const STAGE_ORDER: Stage[] = ['uploading', 'pass1', 'pass2', 'rendering', 'complete']

interface Props {
  jobId: string
  onComplete: (data: ResultsData) => void
  onCancel: () => void
}

export default function ProcessingScreen({ jobId, onComplete, onCancel }: Props) {
  const [stage, setStage] = useState<Stage>('pass1')
  const [progress, setProgress] = useState(5)
  const [logLines, setLogLines] = useState<string[]>([])
  const logRef = useRef<HTMLPreElement>(null)
  const sseRef = useRef<EventSource | null>(null)

  useEffect(() => {
    const es = new EventSource(`/api/status/${jobId}`)
    sseRef.current = es

    es.onmessage = (e) => {
      const data = JSON.parse(e.data) as { stage: Stage; progress: number; message: string }
      setStage(data.stage)
      setProgress(data.progress)
      if (data.message) setLogLines(prev => [...prev.slice(-99), data.message])

      if (data.stage === 'complete') {
        es.close()
        fetch(`/api/results/${jobId}`)
          .then(r => r.json())
          .then((rd: { fps: number; total_frames: number; report: ResultsData['report'] }) => {
            onComplete({ fps: rd.fps, total_frames: rd.total_frames, report: rd.report })
          })
          .catch(err => {
            setLogLines(prev => [...prev, `Error loading results: ${err}`])
            setStage('error')
          })
      }
      if (data.stage === 'error') {
        es.close()
      }
    }

    es.onerror = () => {
      setStage('error')
      es.close()
    }

    return () => es.close()
  }, [jobId, onComplete])

  // Auto-scroll log
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight
    }
  }, [logLines])

  function cancel() {
    sseRef.current?.close()
    onCancel()
  }

  const currentStageIdx = STAGE_ORDER.indexOf(stage)

  return (
    <div className="min-h-screen bg-slate-950 flex flex-col items-center justify-center p-8">
      <div className="w-full max-w-xl">
        {/* Header */}
        <div className="flex items-center gap-3 mb-10">
          <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center">
            <Zap className="w-5 h-5 text-white" />
          </div>
          <span className="text-white font-semibold text-lg">NeuriveAI</span>
        </div>

        <h2 className="text-2xl font-bold text-white mb-2">Analyzing video…</h2>
        <p className="text-slate-400 text-sm mb-8">
          Two-pass pipeline: fast scan + clinical brain analysis
        </p>

        {/* Stage pills */}
        <div className="flex items-center gap-2 mb-6 flex-wrap">
          {STAGE_ORDER.filter(s => s !== 'complete').map((s, i) => {
            const idx = STAGE_ORDER.indexOf(s)
            const isDone = currentStageIdx > idx
            const isActive = stage === s
            return (
              <span
                key={s}
                className={`
                  px-3 py-1.5 rounded-full text-xs font-semibold transition-colors
                  ${isActive  ? 'bg-blue-600 text-white'
                  : isDone    ? 'bg-green-700 text-green-100'
                              : 'bg-slate-800 text-slate-500'}
                `}
              >
                {i + 1}. {STAGE_LABELS[s]}
              </span>
            )
          })}
        </div>

        {/* Progress bar */}
        <div className="w-full bg-slate-800 rounded-full h-2 mb-2 overflow-hidden">
          <div
            className="h-2 rounded-full transition-all duration-700 bg-blue-500"
            style={{ width: `${progress}%` }}
          />
        </div>
        <p className="text-slate-500 text-xs text-right mb-6">{progress}%</p>

        {/* Log */}
        <pre
          ref={logRef}
          className="
            w-full bg-slate-900 border border-slate-800 text-slate-400 text-xs font-mono
            rounded-xl p-4 h-48 overflow-y-auto scrollbar-thin whitespace-pre-wrap break-all
          "
        >
          {logLines.join('\n') || 'Waiting for output…'}
        </pre>

        {/* Error state */}
        {stage === 'error' && (
          <div className="mt-4 p-3 bg-red-950/40 border border-red-800 rounded-lg text-red-400 text-sm">
            Pipeline error. Check the log above for details.
          </div>
        )}

        {/* Cancel */}
        <div className="mt-6 flex justify-center">
          <button
            onClick={cancel}
            className="flex items-center gap-2 text-slate-500 hover:text-red-400 text-sm transition-colors"
          >
            <X className="w-4 h-4" />
            Cancel
          </button>
        </div>
      </div>
    </div>
  )
}
