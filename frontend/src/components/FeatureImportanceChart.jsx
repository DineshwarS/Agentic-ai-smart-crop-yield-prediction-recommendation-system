import React from 'react'

export default function FeatureImportanceChart({ importances, title = 'Model Feature Importance' }) {
  if (!importances || Object.keys(importances).length === 0) {
    return null
  }

  // Sort and take top 8
  const sorted = Object.entries(importances)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8)

  const maxVal = Math.max(...sorted.map(([, v]) => v)) || 1

  return (
    <div className="card">
      <div className="card-title">{title}</div>
      <div className="card-subtitle mb-4">
        Factors influencing the prediction model (SHAP / Gini Importance)
      </div>

      <div>
        {sorted.map(([feature, val]) => {
          const pct = Math.round((val / maxVal) * 100)
          return (
            <div key={feature} className="feat-bar-item">
              <div className="feat-bar-label">
                <span style={{ textTransform: 'capitalize' }}>
                  {feature.replace(/_/g, ' ')}
                </span>
                <span className="font-mono">{val.toFixed(4)}</span>
              </div>
              <div className="feat-bar-track">
                <div
                  className="feat-bar-fill"
                  style={{ width: `${pct}%` }}
                ></div>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
