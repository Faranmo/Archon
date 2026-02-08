import React from 'react'

interface Props {
  value: string
  onChange: (value: string) => void
  onSubmit: () => void
  loading: boolean
}

export function ResearchInput({ value, onChange, onSubmit, loading }: Props) {
  return (
    <div className="bg-white rounded-lg shadow p-4">
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="Enter your research query..."
        className="w-full h-32 p-3 border rounded-lg resize-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
        disabled={loading}
      />
      <div className="flex justify-end mt-3">
        <button
          onClick={onSubmit}
          disabled={loading || !value.trim()}
          className="px-6 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
        >
          {loading ? (
            <>
              <span className="animate-spin">⟳</span>
              Researching...
            </>
          ) : (
            'Start Research'
          )}
        </button>
      </div>
    </div>
  )
}
