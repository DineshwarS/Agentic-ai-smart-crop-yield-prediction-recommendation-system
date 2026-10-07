import React, { useState, useEffect } from 'react'
import { Sprout, Cloud, RefreshCw, AlertCircle, CheckCircle2, Info } from 'lucide-react'
import Header from '../components/Header'
import FeatureImportanceChart from '../components/FeatureImportanceChart'
import { getCrops, getStates, getSeasons, getStateSoil, getWeather, predictYield } from '../api'

const STATE_COORDINATES = {
  'Andhra Pradesh': { lat: 16.3067, lon: 80.4365 },
  'Assam': { lat: 26.1445, lon: 91.7362 },
  'Bihar': { lat: 25.5941, lon: 85.1376 },
  'Chhattisgarh': { lat: 21.2514, lon: 81.6296 },
  'Gujarat': { lat: 23.2156, lon: 72.6369 },
  'Haryana': { lat: 29.0588, lon: 76.0856 },
  'Karnataka': { lat: 15.3173, lon: 75.7139 },
  'Kerala': { lat: 10.8505, lon: 76.2711 },
  'Madhya Pradesh': { lat: 22.9734, lon: 78.6569 },
  'Maharashtra': { lat: 19.7515, lon: 75.7139 },
  'Odisha': { lat: 20.9517, lon: 85.0985 },
  'Punjab': { lat: 31.1471, lon: 75.3412 },
  'Rajasthan': { lat: 27.0238, lon: 74.2179 },
  'Tamil Nadu': { lat: 11.1271, lon: 78.6569 },
  'Telangana': { lat: 18.1124, lon: 79.0193 },
  'Uttar Pradesh': { lat: 26.8467, lon: 80.9462 },
  'West Bengal': { lat: 22.9868, lon: 87.8550 },
}

export default function YieldPrediction() {
  const [crops, setCrops] = useState([])
  const [states, setStates] = useState([])
  const [seasons, setSeasons] = useState([])

  const [formData, setFormData] = useState({
    crop: 'Rice',
    state: 'Punjab',
    season: 'Kharif',
    area: 10,
    fertilizer: 1500,
    pesticide: 25,
    year: 2024,
    N: 150,
    P: 50,
    K: 40,
    pH: 8.0,
    useLiveWeather: true,
    avg_temp_c: 24.5,
    total_rainfall_mm: 850.0,
    avg_humidity_percent: 65.0,
  })

  const [liveWeather, setLiveWeather] = useState(null)
  const [fetchingWeather, setFetchingWeather] = useState(false)
  const [predicting, setPredicting] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    loadDropdowns()
  }, [])

  useEffect(() => {
    if (formData.state) {
      handleStateChange(formData.state)
    }
  }, [formData.state])

  const loadDropdowns = async () => {
    try {
      const [cropsRes, statesRes, seasonsRes] = await Promise.all([
        getCrops(),
        getStates(),
        getSeasons(),
      ])
      setCrops(cropsRes.crops || [])
      setStates(statesRes.states || [])
      setSeasons(seasonsRes.seasons || [])
    } catch (err) {
      console.error('Failed to load dropdown options:', err)
    }
  }

  const handleStateChange = async (stateName) => {
    // 1. Fetch soil defaults for state
    try {
      const soil = await getStateSoil(stateName)
      setFormData(prev => ({
        ...prev,
        state: stateName,
        N: soil.N ?? prev.N,
        P: soil.P ?? prev.P,
        K: soil.K ?? prev.K,
        pH: soil.pH ?? prev.pH,
      }))
    } catch (e) {
      console.warn('No default soil for state:', stateName)
    }

    // 2. Fetch live weather for state coordinates
    const coords = STATE_COORDINATES[stateName]
    if (coords) {
      setFetchingWeather(true)
      try {
        const w = await getWeather(coords.lat, coords.lon, stateName)
        setLiveWeather(w)
        if (w.annual_est_temp_c != null) {
          setFormData(prev => ({
            ...prev,
            avg_temp_c: w.annual_est_temp_c,
            total_rainfall_mm: w.annual_est_rainfall_mm,
            avg_humidity_percent: w.annual_est_humidity_pct,
          }))
        }
      } catch (err) {
        console.error('Failed to fetch weather for state:', err)
      } finally {
        setFetchingWeather(false)
      }
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setPredicting(true)
    setError(null)
    setResult(null)

    try {
      const payload = {
        crop: formData.crop,
        state: formData.state,
        season: formData.season,
        area: Number(formData.area),
        fertilizer: Number(formData.fertilizer),
        pesticide: Number(formData.pesticide),
        year: Number(formData.year),
        N: Number(formData.N),
        P: Number(formData.P),
        K: Number(formData.K),
        pH: Number(formData.pH),
        avg_temp_c: Number(formData.avg_temp_c),
        total_rainfall_mm: Number(formData.total_rainfall_mm),
        avg_humidity_percent: Number(formData.avg_humidity_percent),
      }

      if (formData.useLiveWeather && STATE_COORDINATES[formData.state]) {
        payload.latitude = STATE_COORDINATES[formData.state].lat
        payload.longitude = STATE_COORDINATES[formData.state].lon
      }

      const res = await predictYield(payload)
      setResult(res)
    } catch (err) {
      setError(err.message || 'Prediction failed')
    } finally {
      setPredicting(false)
    }
  }

  return (
    <div className="main-content">
      <Header
        title="Crop Yield Prediction"
        subtitle="Estimate regional crop yield (tonnes/ha) with Leakage-Free XGBoost / Random Forest regression"
      />

      <div className="page-content">
        <div className="grid-2">
          {/* Input Form */}
          <div className="card">
            <div className="card-title mb-4">Input Parameters</div>

            {error && (
              <div className="alert alert-error">
                <AlertCircle size={16} />
                <span>{error}</span>
              </div>
            )}

            <form onSubmit={handleSubmit}>
              <div className="form-grid">
                <div className="form-group">
                  <label>State / Region</label>
                  <select
                    value={formData.state}
                    onChange={e => setFormData({ ...formData, state: e.target.value })}
                  >
                    {states.map(s => (
                      <option key={s} value={s}>{s}</option>
                    ))}
                  </select>
                </div>

                <div className="form-group">
                  <label>Crop</label>
                  <select
                    value={formData.crop}
                    onChange={e => setFormData({ ...formData, crop: e.target.value })}
                  >
                    {crops.map(c => (
                      <option key={c} value={c}>{c}</option>
                    ))}
                  </select>
                </div>

                <div className="form-group">
                  <label>Season</label>
                  <select
                    value={formData.season}
                    onChange={e => setFormData({ ...formData, season: e.target.value })}
                  >
                    {seasons.map(s => (
                      <option key={s} value={s}>{s}</option>
                    ))}
                  </select>
                </div>

                <div className="form-group">
                  <label>Area (Hectares)</label>
                  <input
                    type="number"
                    step="0.1"
                    min="0.1"
                    value={formData.area}
                    onChange={e => setFormData({ ...formData, area: e.target.value })}
                    required
                  />
                </div>

                <div className="form-group">
                  <label>Total Fertilizer (kg)</label>
                  <input
                    type="number"
                    min="0"
                    value={formData.fertilizer}
                    onChange={e => setFormData({ ...formData, fertilizer: e.target.value })}
                    required
                  />
                </div>

                <div className="form-group">
                  <label>Pesticide (kg)</label>
                  <input
                    type="number"
                    step="0.1"
                    min="0"
                    value={formData.pesticide}
                    onChange={e => setFormData({ ...formData, pesticide: e.target.value })}
                    required
                  />
                </div>

                <div className="form-group">
                  <label>Prediction Year</label>
                  <input
                    type="number"
                    min="1997"
                    max="2030"
                    value={formData.year}
                    onChange={e => setFormData({ ...formData, year: e.target.value })}
                    required
                  />
                </div>

                <div className="form-group">
                  <label>Soil pH</label>
                  <input
                    type="number"
                    step="0.1"
                    min="0"
                    max="14"
                    value={formData.pH}
                    onChange={e => setFormData({ ...formData, pH: e.target.value })}
                    required
                  />
                </div>

                <div className="form-group">
                  <label>Soil Nitrogen (N)</label>
                  <input
                    type="number"
                    value={formData.N}
                    onChange={e => setFormData({ ...formData, N: e.target.value })}
                    required
                  />
                </div>

                <div className="form-group">
                  <label>Soil Phosphorus (P)</label>
                  <input
                    type="number"
                    value={formData.P}
                    onChange={e => setFormData({ ...formData, P: e.target.value })}
                    required
                  />
                </div>

                <div className="form-group">
                  <label>Soil Potassium (K)</label>
                  <input
                    type="number"
                    value={formData.K}
                    onChange={e => setFormData({ ...formData, K: e.target.value })}
                    required
                  />
                </div>
              </div>

              <div className="divider"></div>

              {/* Weather info */}
              <div className="flex-between mb-3">
                <label className="flex gap-2" style={{ alignItems: 'center' }}>
                  <Cloud size={16} className="text-accent" />
                  <span>Climate / Weather Parameters</span>
                </label>
                {fetchingWeather && <span className="text-xs text-muted">Updating live weather...</span>}
              </div>

              <div className="form-grid">
                <div className="form-group">
                  <label>Annual Temp (°C)</label>
                  <input
                    type="number"
                    step="0.1"
                    value={formData.avg_temp_c}
                    onChange={e => setFormData({ ...formData, avg_temp_c: e.target.value })}
                  />
                </div>

                <div className="form-group">
                  <label>Annual Rainfall (mm)</label>
                  <input
                    type="number"
                    step="1"
                    value={formData.total_rainfall_mm}
                    onChange={e => setFormData({ ...formData, total_rainfall_mm: e.target.value })}
                  />
                </div>

                <div className="form-group">
                  <label>Annual Humidity (%)</label>
                  <input
                    type="number"
                    step="0.1"
                    value={formData.avg_humidity_percent}
                    onChange={e => setFormData({ ...formData, avg_humidity_percent: e.target.value })}
                  />
                </div>
              </div>

              <div className="mt-4">
                <button
                  type="submit"
                  className="btn btn-primary btn-lg w-full flex-center"
                  disabled={predicting}
                >
                  {predicting ? (
                    <>
                      <div className="spinner"></div>
                      <span>Running Model Inference...</span>
                    </>
                  ) : (
                    <>
                      <Sprout size={18} />
                      <span>Calculate Expected Yield</span>
                    </>
                  )}
                </button>
              </div>
            </form>
          </div>

          {/* Results & Explainability */}
          <div>
            {result ? (
              <div className="flex-col gap-4">
                <div className="result-panel">
                  <div className="badge badge-green mb-2">Prediction Generated</div>
                  <div className="result-yield">
                    {result.predicted_yield_tonnes_ha}{' '}
                    <span className="result-unit">tonnes / hectare</span>
                  </div>
                  <p className="text-sm text-muted mb-4">
                    Target Total Production on {formData.area} ha: <strong>{(result.predicted_yield_tonnes_ha * formData.area).toFixed(2)} tonnes</strong>
                  </p>

                  <div className="grid-2 text-sm pt-4" style={{ borderTop: '1px solid var(--border-subtle)' }}>
                    <div>
                      <span className="text-muted">Selected Model:</span>{' '}
                      <strong>{result.model_used}</strong>
                    </div>
                    <div>
                      <span className="text-muted">Weather Source:</span>{' '}
                      <span className="badge badge-cyan">{result.weather_source}</span>
                    </div>
                  </div>
                </div>

                <FeatureImportanceChart
                  importances={result.feature_importance}
                  title="Feature Influences on this Yield Model"
                />

                <div className="card">
                  <div className="card-title flex gap-2" style={{ alignItems: 'center' }}>
                    <Info size={16} className="text-accent" />
                    <span>Scientific Disclaimer</span>
                  </div>
                  <p className="text-xs text-muted mt-2">
                    {result.disclaimer}
                  </p>
                </div>
              </div>
            ) : (
              <div className="card empty-state">
                <Sprout size={48} className="empty-icon text-primary-green" />
                <h3>No Prediction Yet</h3>
                <p>
                  Fill in your farm parcel details, select your state and crop, and click
                  <strong> "Calculate Expected Yield"</strong>.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
