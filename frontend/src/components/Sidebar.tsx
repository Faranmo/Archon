import React from 'react'

export function Sidebar() {
  return (
    <aside className="w-64 bg-gray-900 text-white p-4">
      <div className="flex items-center gap-2 mb-8">
        <span className="text-2xl">🔬</span>
        <span className="text-xl font-bold">Archon</span>
      </div>
      
      <nav className="space-y-2">
        <a href="#" className="block px-4 py-2 rounded-lg bg-gray-800 hover:bg-gray-700">
          Research
        </a>
        <a href="#" className="block px-4 py-2 rounded-lg hover:bg-gray-800">
          Documents
        </a>
        <a href="#" className="block px-4 py-2 rounded-lg hover:bg-gray-800">
          History
        </a>
        <a href="#" className="block px-4 py-2 rounded-lg hover:bg-gray-800">
          Settings
        </a>
      </nav>
      
      <div className="absolute bottom-4 left-4 right-4">
        <div className="bg-gray-800 rounded-lg p-3 text-sm">
          <p className="text-gray-400">Budget</p>
          <div className="flex justify-between items-center mt-1">
            <span className="font-semibold">$4.50 / $10.00</span>
            <span className="text-green-400">45%</span>
          </div>
        </div>
      </div>
    </aside>
  )
}
