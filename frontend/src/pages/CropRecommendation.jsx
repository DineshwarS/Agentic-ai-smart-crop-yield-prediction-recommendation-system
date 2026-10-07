import React, { useState, useEffect } from 'react'
import { Sparkles, CheckCircle2, AlertCircle, Info, RefreshCw, Layers } from 'lucide-react'
import Header from '../components/Header'
import { getStates, getStateSoil, getWeather, recommendCrop } from '../api'

const STATE_COORDS = {
  'Punjab': { lat: 31.1471, lon: 75.3412 },
  'Haryana': { lat: 29.0588, lon: 76.0856 },
  'Maharashtra': { lat: 19.7515, lon: 75.7139 },
  'Andhra Pradesh': { lat: 16.3067, lon: 80.4365 },
  'Tamil Nadu': { lat: 11.1271, lon: 78.6569 },
  'Karnataka': { lat: 15.3173, lon: 75.7139 },
  'Uttar Pradesh': { lat: 26.8467, lon: 80.9462 },
  'Madhya Pradesh': { lat: 22.9734, lon: 78.6569 },
  'Gujarat': { lat: 23.2156, lon: 72.6369 },
  'Bihar': { lat: 25.5941, lon: 85.1376 },
  'West Bengal': { lat: 22.9868, lon: 87.8550 },
}

export default function CropRecommendation() {
  const [states, setStates] = useState([])
  const [selectedState, setSelectedState] = useState('Punjab')

  const [formData, setFormData] = useState({
    N: 90,
    P: 42,
    K: 43,
    ph: 6.5,
    temperature: 24.5,
    humidity: 80.0,
    rainfall: 180.0,
  })

  const [loadingWeather, setLoadingWeather] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    loadStates()
  }, [])

  useEffect(() => {
    if (selectedState) {
      handleStateChange(selectedState)
    }
  }, [selectedState])

  const loadStates = async () => {
    try {
      const res = await getStates()
      setStates(res.states || [])
    } catch (e) {
      console.error(e)
    }
  }

  const handleStateChange = async (st) => {
    setSelectedState(st)
    // 1. Soil defaults
    try {
      const soil = await getStateSoil(st)
      setFormData(prev => ({
        ...prev,
        N: soil.N ?? prev.N,
        P: soil.P ?? prev.P,
        K: soil.K ?? prev.K,
        ph: soil.pH ?? prev.ph,
      }))
    } catch (e) {}

    // 2. Weather
    const coords = STATE_COORDS[st]
    if (coords) {
      setLoadingWeather(true)
      try {
        const w = await getWeather(coords.lat, coords.lon, st)
        setFormData(prev => ({
          ...prev,
          temperature: w.current_temperature_c ?? prev.temperature,
          humidity: w.current_humidity_pct ?? prev.humidity,
          rainfall: w.current_precipitation_mm != null ? Math.max(w.current_precipitation_mm * 30, 80) : prev.rainfall,
        }))
      } catch (err) {
        console.error(err)
      } finally {
        setLoadingWeather(false)
      }
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setSubmitting(true)
    setError(null)
    setResult(null)

    try {
      const payload = {
        N: Number(formData.N),
        P: Number(formData.P),
        K: Number(formData.K),
        ph: Number(formData.ph),
        temperature: Number(formData.temperature),
        humidity: Number(formData.humidity),
        rainfall: Number(formData.rainfall),
        state: selectedState,
      }
      const res = await recommendCrop(payload)
      setResult(res)
    } catch (err) {
      setError(err.message || 'Crop recommendation failed')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="main-content">
      <Header
        title="Crop Recommendation Engine"
        subtitle="Multi-class classification evaluating soil chemistry and micro-climatic thresholds to find optimal crops"
      />

      <div className="page-content">
        <div className="grid-2">
          {/* Form */}
          <div className="card">
            <div className="card-title mb-4">Soil & Agro-Climatic Profile</div>

            {error && (
              <div className="alert alert-error">
                <AlertCircle size={16} />
                <span>{error}</span>
              </div>
            )}

            <form onSubmit={handleSubmit}>
              <div className="form-group mb-4">
                <label>Region / State (Optional Pre-fill)</label>
                <select
                  value={selectedState}
                  onChange={e => handleStateChange(e.target.value)}
                >
                  {states.map(s => (
                    <option key={s} value={s}>{s}</option>
                  ))}
                </select>
              </div>

              <div className="form-grid">
                <div className="form-group">
                  <label>Nitrogen (N) ratio</label>
                  <input
                    type="number"
                    value={formData.N}
                    onChange={e => setFormData({ ...formData, N: e.target.value })}
                    required
                  />
                </div>

                <div className="form-group">
                  <label>Phosphorus (P) ratio</label>
                  <input
                    type="number"
                    value={formData.P}
                    onChange={e => setFormData({ ...formData, P: e.target.value })}
                    required
                  />
                </div>

                <div className="form-group">
                  <label>Potassium (K) ratio</label>
                  <input
                    type="number"
                    value={formData.K}
                    onChange={e => setFormData({ ...formData, K: e.target.value })}
                    required
                  />
                </div>

                <div className="form-group">
                  <label>Soil pH level</label>
                  <input
                    type="number"
                    step="0.1"
                    min="0"
                    max="14"
                    value={formData.ph}
                    onChange={e => setFormData({ ...formData, ph: e.target.value })}
                    required
                  />
                </div>

                <div className="form-group">
                  <label>Temperature (°C)</label>
                  <input
                    type="number"
                    step="0.1"
                    value={formData.temperature}
                    onChange={e => setFormData({ ...formData, temperature: e.target.value })}
                    required
                  />
                </div>

                <div className="form-group">
                  <label>Relative Humidity (%)</label>
                  <input
                    type="number"
                    step="0.1"
                    value={formData.humidity}
                    onChange={e => setFormData({ ...formData, humidity: e.target.value })}
                    required
                  />
                </div>

                <div className="form-group full-width">
                  <label>Growing Season Rainfall (mm)</label>
                  <input
                    type="number"
                    step="1"
                    value={formData.rainfall}
                    onChange={e => setFormData({ ...formData, rainfall: e.target.value })}
                    required
                  />
                </div>
              </div>

              <div className="mt-4">
                <button
                  type="submit"
                  className="btn btn-primary btn-lg w-full flex-center"
                  disabled={submitting}
                >
                  {submitting ? (
                    <>
                      <div className="spinner"></div>
                      <span>Evaluating Soil Compatibility...</span>
                    </>
                  ) : (
                    <>
                      <Sparkles size={18} />
                      <span>Find Best Suited Crop</span>
                    </>
                  )}
                </button>
              </div>
            </form>
          </div>

          {/* Results */}
          <div>
            {result ? (
              <div className="flex-col gap-4">
                <div className="result-panel">
                  <div className="badge badge-cyan mb-2">Primary Recommendation</div>
                  <div
                    className="result-yield"
                    style={{ textTransform: 'capitalize', fontSize: 42 }}
                  >
                    {result.recommended_crop}
                  </div>
                  <p className="text-sm mt-3" style={{ lineHeight: 1.6 }}>
                    {result.explanation}
                  </p>
                  <div className="mt-4 text-xs text-muted">
                    Evaluated by: <strong>{result.model_used}</strong>
                  </div>
                </div>

                {result.top_crops && result.top_crops.length > 1 && (
                  <div className="card">
                    <div className="card-title mb-1">Alternative Suitable Crops</div>
                    <div className="card-subtitle mb-3">Model Confidence Probabilities</div>

                    <div>
                      {result.top_crops.map((cropObj, idx) => (
                        <div key={cropObj.crop} className="crop-rank-item">
                          <div className={`rank-number ${idx === 0 ? 'first' : ''}`}>
                            #{idx + 1}
                          </div>
                          <div className="rank-crop" style={{ textTransform: 'capitalize' }}>
                            {cropObj.crop}
                          </div>
                          <div className="rank-score font-mono">
                            {(cropObj.score * 100).toFixed(1)}%
                          </div>
                          <div className="rank-bar">
                            <div
                              className="rank-bar-fill"
                              style={{ width: `${Math.round(cropObj.score * 100)}%` }}
                            ></div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="card empty-state">
                <Sparkles size={48} className="empty-icon text-accent" />
                <h3>No Recommendation Yet</h3>
                <p>
                  Input your soil chemistry (N-P-K-pH) and climatic metrics to receive 
                  an optimal crop classification.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
