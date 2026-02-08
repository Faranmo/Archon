import { useState } from 'react'
import { api } from '../services/api'

interface ResearchResult {
  success: boolean
  output?: string
  plan?: any
  research?: any
  analysis?: any
  iterations: number
  error?: string
}

export function useResearch() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<ResearchResult | null>(null)

  const research = async (query: string) => {
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await api.post('/research', {
        query,
        depth: 'moderate',
        max_iterations: 10,
      })
      setResult(response.data)
    } catch (err: any) {
      setError(err.response?.data?.error || 'Research failed')
    } finally {
      setLoading(false)
    }
  }

  return { research, loading, error, result }
}
