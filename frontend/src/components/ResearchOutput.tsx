import React from 'react'
import ReactMarkdown from 'react-markdown'

interface ResearchResult {
  success: boolean
  output?: string
  plan?: any
  research?: any
  analysis?: any
  iterations: number
  error?: string
}

interface Props {
  result: ResearchResult
}

export function ResearchOutput({ result }: Props) {
  if (!result.success) {
    return (
      <div className="mt-6 p-4 bg-red-50 text-red-700 rounded-lg">
        <h3 className="font-semibold">Research Failed</h3>
        <p>{result.error}</p>
      </div>
    )
  }

  return (
    <div className="mt-6 space-y-6">
      {/* Main Output */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-bold mb-4">Research Report</h2>
        <div className="prose max-w-none">
          <ReactMarkdown>{result.output || ''}</ReactMarkdown>
        </div>
      </div>

      {/* Metadata */}
      <div className="bg-gray-100 rounded-lg p-4 text-sm text-gray-600">
        <p>Completed in {result.iterations} iterations</p>
      </div>
    </div>
  )
}
