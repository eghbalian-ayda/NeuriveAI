import { useEffect, useRef, useState } from 'react'
import { X } from 'lucide-react'
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
      if (data.message) setLogLines(prev => {
        if (prev.length > 0 && prev[prev.length - 1] === data.message) return prev
        return [...prev.slice(-99), data.message]
      })

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
    <div className="min-h-screen bg-gray-50 flex flex-col items-center justify-center p-8">
      <div className="w-full max-w-xl">
        {/* Header */}
        <div className="flex items-center gap-3 mb-10">
          <img src="/logo.png" alt="NeuriveAI logo" className="w-8 h-8 sm:w-12 sm:h-12 object-contain" />
          <span className="text-slate-900 font-semibold text-lg sm:text-2xl">NeuriveAI</span>
        </div>

        <h2 className="text-2xl sm:text-4xl font-bold text-slate-900 mb-2">Analyzing video…</h2>
        <p className="text-slate-500 text-sm sm:text-xl mb-8">
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
                  px-3 py-1.5 sm:px-5 sm:py-3 rounded-full text-xs sm:text-base font-semibold transition-colors
                  ${isActive  ? 'bg-blue-600 text-white'
                  : isDone    ? 'bg-green-100 text-green-700'
                              : 'bg-slate-200 text-slate-400'}
                `}
              >
                {i + 1}. {STAGE_LABELS[s]}
              </span>
            )
          })}
        </div>

        {/* Progress bar */}
        <div className="w-full bg-slate-200 rounded-full h-2 mb-2 overflow-hidden">
          <div
            className="h-2 rounded-full transition-all duration-700 bg-blue-500"
            style={{ width: `${progress}%` }}
          />
        </div>
        <p className="text-slate-400 text-xs sm:text-base text-right mb-6">{progress}%</p>

        {/* Log */}
        <pre
          ref={logRef}
          className="
            w-full bg-white border border-slate-200 text-slate-600 text-xs sm:text-base font-mono
            rounded-xl p-4 h-48 sm:h-60 overflow-y-auto scrollbar-thin whitespace-pre-wrap break-all
          "
        >
          {logLines.join('\n') || 'Waiting for output…'}
        </pre>

        {/* Error state */}
        {stage === 'error' && (
          <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-600 text-sm sm:text-xl">
            Pipeline error. Check the log above for details.
          </div>
        )}

        {/* Cancel */}
        <div className="mt-6 flex justify-center">
          <button
            onClick={cancel}
            className="flex items-center gap-2 text-slate-400 hover:text-red-500 text-sm sm:text-xl transition-colors"
          >
            <X className="w-4 h-4 sm:w-6 sm:h-6" />
            Cancel
          </button>
        </div>
      </div>
    </div>
  )
}
