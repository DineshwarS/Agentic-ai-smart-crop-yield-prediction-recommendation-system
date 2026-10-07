import React from 'react'
import { Sparkles, ShieldCheck } from 'lucide-react'

export default function Header({ title, subtitle }) {
  return (
    <header className="page-header flex-between">
      <div>
        <h1 className="page-title">{title}</h1>
        {subtitle && <p className="page-subtitle">{subtitle}</p>}
      </div>
      <div className="flex gap-3">
        <div className="badge badge-green">
          <ShieldCheck size={12} />
          <span>Zero IoT | Pure ML & Agentic AI</span>
        </div>
      </div>
    </header>
  )
}
