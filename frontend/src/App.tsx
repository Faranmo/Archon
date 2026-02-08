import React, { useState } from 'react'
import { ResearchInput } from './components/ResearchInput'
import { ResearchOutput } from './components/ResearchOutput'
import { Sidebar } from './components/Sidebar'
import { useResearch } from './hooks/useResearch'

function App() {
  const { research, loading, error, result } = useResearch()
  const [query, setQuery] = useState('')

  const handleSubmit = async () => {
    if (query.trim()) {
      await research(query)
    }
  }

  return (
    <div className="flex h-screen">
      <Sidebar />
      <main className="flex-1 flex flex-col">
        <header className="bg-white border-b px-6 py-4">
          <h1 className="text-2xl font-bold text-gray-900">Archon</h1>
          <p className="text-sm text-gray-500">AI Research Assistant</p>
        </header>
        
        <div className="flex-1 overflow-y-auto p-6">
          <ResearchInput
            value={query}
            onChange={setQuery}
            onSubmit={handleSubmit}
            loading={loading}
          />
          
          {error && (
            <div className="mt-4 p-4 bg-red-50 text-red-700 rounded-lg">
              {error}
            </div>
          )}
          
          {result && (
            <ResearchOutput result={result} />
          )}
        </div>
      </main>
    </div>
  )
}

export default App
