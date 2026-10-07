import React, { useState, useEffect } from 'react'
import { History, RefreshCw, Layers, Calendar, MapPin, Sprout } from 'lucide-react'
import Header from '../components/Header'
import { getPredictions } from '../api'

export default function PredictionHistory() {
  const [predictions, setPredictions] = useState([])
  const [loading, setLoading] = useState(true)
  const [filterType, setFilterType] = useState('all')

  const loadData = async () => {
    setLoading(true)
    try {
      const res = await getPredictions(100)
      setPredictions(res.predictions || [])
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadData()
  }, [])

  const filtered = predictions.filter(p => {
    if (filterType === 'yield') return p.type === 'yield'
    if (filterType === 'recommendation') return p.type === 'recommendation'
    return true
  })

  return (
    <div className="main-content">
      <Header
        title="Prediction & Advisory History"
        subtitle="Auditable log of past crop yield forecasts and crop suitability recommendations persisted to MongoDB"
      />

      <div className="page-content">
        <div className="card">
          <div className="flex-between mb-4" style={{ flexWrap: 'wrap', gap: 12 }}>
            <div className="tabs">
              <button
                className={`tab-btn ${filterType === 'all' ? 'active' : ''}`}
                onClick={() => setFilterType('all')}
              >
                All Records ({predictions.length})
              </button>
              <button
                className={`tab-btn ${filterType === 'yield' ? 'active' : ''}`}
                onClick={() => setFilterType('yield')}
              >
                Yield Forecasts
              </button>
              <button
                className={`tab-btn ${filterType === 'recommendation' ? 'active' : ''}`}
                onClick={() => setFilterType('recommendation')}
              >
                Crop Recommendations
              </button>
            </div>

            <button className="btn btn-ghost btn-sm flex gap-1" onClick={loadData}>
              <RefreshCw size={13} />
              <span>Refresh</span>
            </button>
          </div>

          {loading ? (
            <div className="loading-overlay">
              <div className="spinner"></div>
              <p>Loading prediction records...</p>
            </div>
          ) : filtered.length === 0 ? (
            <div className="empty-state">
              <History size={48} className="empty-icon" />
              <h3>No Prediction Records</h3>
              <p>
                Run yield predictions or crop recommendations from the navigation menu to see your saved history.
              </p>
            </div>
          ) : (
            <div className="table-wrapper">
              <table>
                <thead>
                  <tr>
                    <th>Timestamp</th>
                    <th>Type</th>
                    <th>Target Crop / Region</th>
                    <th>Prediction Output</th>
                    <th>Model Used</th>
                    <th>Weather Mode</th>
                  </tr>
                </thead>
                <tbody>
                  {filtered.map((item, idx) => {
                    const dateStr = item.timestamp ? new Date(item.timestamp).toLocaleString() : '--'
                    return (
                      <tr key={item._id || idx}>
                        <td className="text-xs text-muted font-mono">{dateStr}</td>
                        <td>
                          <span
                            className={`badge ${item.type === 'yield' ? 'badge-green' : 'badge-cyan'}`}
                          >
                            {item.type}
                          </span>
                        </td>
                        <td>
                          <strong>{item.crop || item.recommended_crop || '--'}</strong>
                          {item.state && <span className="text-muted text-xs"> ({item.state})</span>}
                        </td>
                        <td className="font-mono">
                          {item.predicted_yield_tonnes_ha != null
                            ? `${item.predicted_yield_tonnes_ha} tonnes/ha`
                            : item.recommended_crop}
                        </td>
                        <td className="text-xs">{item.model_used || '--'}</td>
                        <td>
                          <span className="badge badge-amber">{item.weather_source || 'live'}</span>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
