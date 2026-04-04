import { useState } from 'react'
import type { ResultsData } from '@/types/impact'
import UploadScreen from '@/components/UploadScreen'
import ProcessingScreen from '@/components/ProcessingScreen'
import ResultsScreen from '@/components/ResultsScreen'

type Phase = 'upload' | 'processing' | 'results'

export default function App() {
  const [phase, setPhase] = useState<Phase>('upload')
  const [jobId, setJobId] = useState<string>('')
  const [resultsData, setResultsData] = useState<ResultsData | null>(null)

  function handleUploaded(id: string) {
    setJobId(id)
    setPhase('processing')
  }

  function handleDemoLoaded(data: ResultsData, id: string) {
    setJobId(id)
    setResultsData(data)
    setPhase('results')
  }

  function handleProcessingComplete(data: ResultsData) {
    setResultsData(data)
    setPhase('results')
  }

  function handleReset() {
    setPhase('upload')
    setJobId('')
    setResultsData(null)
  }

  return (
    <>
      {phase === 'upload' && (
        <UploadScreen
          onUploaded={handleUploaded}
          onDemoLoaded={handleDemoLoaded}
        />
      )}
      {phase === 'processing' && (
        <ProcessingScreen
          jobId={jobId}
          onComplete={handleProcessingComplete}
          onCancel={handleReset}
        />
      )}
      {phase === 'results' && resultsData && (
        <ResultsScreen
          jobId={jobId}
          data={resultsData}
          onReset={handleReset}
        />
      )}
    </>
  )
}
