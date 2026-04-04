import { useRef, useState } from 'react'
import { Upload, Zap } from 'lucide-react'
import type { ResultsData } from '@/types/impact'

interface Props {
  onUploaded: (jobId: string) => void
  onDemoLoaded: (data: ResultsData, jobId: string) => void
}

export default function UploadScreen({ onUploaded, onDemoLoaded }: Props) {
  const [dragOver, setDragOver] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [loadingDemo, setLoadingDemo] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  async function startUpload(file: File) {
    setUploading(true)
    setError(null)
    try {
      const fd = new FormData()
      fd.append('file', file)
      const res = await fetch('/api/upload', { method: 'POST', body: fd })
      if (!res.ok) throw new Error(`Upload failed: ${res.statusText}`)
      const { job_id } = await res.json()
      onUploaded(job_id)
    } catch (e) {
      setError(String(e))
      setUploading(false)
    }
  }

  async function loadDemo() {
    setLoadingDemo(true)
    setError(null)
    try {
      const res = await fetch('/api/demo')
      if (!res.ok) throw new Error('Demo data not available')
      const data = await res.json()
      onDemoLoaded(
        { fps: data.fps, total_frames: data.total_frames, report: data.report },
        data.job_id,
      )
    } catch (e) {
      setError(String(e))
      setLoadingDemo(false)
    }
  }

  function onDrop(e: React.DragEvent) {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) startUpload(file)
  }

  function onFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (file) startUpload(file)
  }

  const busy = uploading || loadingDemo

  return (
    <div className="min-h-screen bg-slate-950 flex flex-col items-center justify-center p-8">
      {/* Branding */}
      <div className="mb-12 text-center">
        <div className="flex items-center justify-center gap-3 mb-3">
          <div className="w-10 h-10 rounded-xl bg-blue-600 flex items-center justify-center">
            <Zap className="w-6 h-6 text-white" />
          </div>
          <h1 className="text-4xl font-bold text-white tracking-tight">NeuriveAI</h1>
        </div>
        <p className="text-slate-400 text-lg">Ordinary cameras. Extraordinary protection.</p>
        <p className="text-slate-600 text-sm mt-1">
          Clinically-validated head impact detection for football
        </p>
      </div>

      {/* Drop zone */}
      <div
        className={`
          w-full max-w-lg border-2 border-dashed rounded-2xl p-16 text-center
          cursor-pointer transition-all duration-200 select-none
          ${dragOver
            ? 'border-blue-500 bg-blue-950/30'
            : 'border-slate-700 bg-slate-900 hover:border-slate-500 hover:bg-slate-800/50'}
          ${busy ? 'pointer-events-none opacity-50' : ''}
        `}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={() => !busy && inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".mp4,.avi,.mov,.mkv"
          className="hidden"
          onChange={onFileChange}
        />
        <Upload className="w-10 h-10 text-slate-500 mx-auto mb-4" />
        {uploading ? (
          <p className="text-blue-400 font-medium">Uploading video…</p>
        ) : (
          <>
            <p className="text-slate-200 text-lg font-medium">Drop your video here</p>
            <p className="text-slate-500 mt-2 text-sm">or click to browse</p>
            <p className="text-slate-600 text-xs mt-3">MP4 · AVI · MOV · MKV</p>
          </>
        )}
      </div>

      {/* Demo button */}
      <button
        onClick={loadDemo}
        disabled={busy}
        className="
          mt-6 px-6 py-3 rounded-xl border border-slate-700 text-slate-400 text-sm
          hover:border-blue-500 hover:text-blue-400 transition-colors disabled:opacity-40
          flex items-center gap-2
        "
      >
        {loadingDemo ? (
          <span>Loading demo…</span>
        ) : (
          <>
            <Zap className="w-4 h-4" />
            Load Example Data
          </>
        )}
      </button>
      <p className="text-slate-600 text-xs mt-2">Instant demo — no processing required</p>

      {error && (
        <p className="mt-4 text-red-400 text-sm bg-red-950/30 border border-red-800 rounded-lg px-4 py-2">
          {error}
        </p>
      )}
    </div>
  )
}
