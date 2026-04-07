import { useRef, useState } from 'react'
import { Upload } from 'lucide-react'

interface Props {
  onUploaded: (jobId: string) => void
}

export default function UploadScreen({ onUploaded }: Props) {
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
      const res = await fetch('/api/demo/run', { method: 'POST' })
      if (!res.ok) throw new Error('Could not start demo pipeline')
      const { job_id } = await res.json()
      onUploaded(job_id)
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
    <div className="min-h-screen bg-gray-50 flex flex-col items-center justify-center p-8">
      {/* Branding */}
      <div className="mb-12 text-center">
        <div className="flex items-center justify-center gap-3 mb-3">
          <img src="/logo.png" alt="NeuriveAI logo" className="w-12 h-12 sm:w-20 sm:h-20 object-contain" />
          <h1 className="text-4xl sm:text-6xl font-bold text-slate-900 tracking-tight">NeuriveAI</h1>
        </div>
        <p className="text-slate-500 text-lg sm:text-2xl">Ordinary cameras. Extraordinary protection.</p>
        <p className="text-slate-400 text-sm sm:text-xl mt-1">
          Clinically-validated head impact detection for football
        </p>
      </div>

      {/* Drop zone */}
      <div
        className={`
          w-full max-w-lg border-2 border-dashed rounded-2xl p-16 sm:p-20 text-center
          cursor-pointer transition-all duration-200 select-none
          ${dragOver
            ? 'border-blue-500 bg-blue-50'
            : 'border-slate-300 bg-white hover:border-slate-400 hover:bg-slate-50'}
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
        <Upload className="w-10 h-10 sm:w-16 sm:h-16 text-slate-400 mx-auto mb-4" />
        {uploading ? (
          <p className="text-blue-500 font-medium sm:text-2xl">Uploading video…</p>
        ) : (
          <>
            <p className="text-slate-700 text-lg sm:text-2xl font-medium">Drop your video here</p>
            <p className="text-slate-400 mt-2 text-sm sm:text-xl">or click to browse</p>
            <p className="text-slate-400 text-xs sm:text-lg mt-3">MP4 · AVI · MOV · MKV</p>
          </>
        )}
      </div>

      {/* Demo button */}
      <button
        onClick={loadDemo}
        disabled={busy}
        className="
          mt-6 px-6 py-3 sm:px-10 sm:py-5 rounded-xl border border-slate-300 text-slate-500 text-sm sm:text-xl
          hover:border-blue-500 hover:text-blue-500 transition-colors disabled:opacity-40
          flex items-center gap-2 bg-white
        "
      >
        {loadingDemo ? (
          <span>Loading demo…</span>
        ) : (
          <>
            Load Example Data
          </>
        )}
      </button>
      <p className="text-slate-400 text-xs sm:text-lg mt-2">Runs the full pipeline on a sample football clip</p>

      {error && (
        <p className="mt-4 text-red-600 text-sm sm:text-xl bg-red-50 border border-red-200 rounded-lg px-4 py-2">
          {error}
        </p>
      )}
    </div>
  )
}
