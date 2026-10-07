import React from 'react'
import { Cloud, Droplets, Thermometer, Wind, RefreshCw } from 'lucide-react'

export default function WeatherWidget({ weather, loading, onRefresh, locationName }) {
  if (loading) {
    return (
      <div className="card loading-overlay">
        <div className="spinner"></div>
        <p>Fetching live weather data from Open-Meteo...</p>
      </div>
    )
  }

  if (!weather) {
    return (
      <div className="card flex-between">
        <div>
          <div className="card-title">Live Weather</div>
          <div className="card-subtitle">No weather data loaded. Select a location.</div>
        </div>
        {onRefresh && (
          <button className="btn btn-secondary btn-sm" onClick={onRefresh}>
            <RefreshCw size={14} /> Fetch Weather
          </button>
        )}
      </div>
    )
  }

  return (
    <div className="card">
      <div className="flex-between mb-4">
        <div>
          <div className="card-title flex gap-2" style={{ alignItems: 'center' }}>
            <Cloud size={18} className="text-accent" />
            <span>Live Regional Weather</span>
          </div>
          <div className="card-subtitle">
            {locationName || weather.location || 'Selected Location'} &bull; Source: Open-Meteo API
          </div>
        </div>
        {onRefresh && (
          <button className="btn btn-ghost btn-sm" onClick={onRefresh} title="Refresh live weather">
            <RefreshCw size={13} />
          </button>
        )}
      </div>

      <div className="weather-grid">
        <div className="weather-item">
          <div className="weather-value">{weather.current_temperature_c ?? '--'}°C</div>
          <div className="weather-label flex-center gap-1">
            <Thermometer size={12} /> Temp
          </div>
        </div>
        <div className="weather-item">
          <div className="weather-value">{weather.current_humidity_pct ?? '--'}%</div>
          <div className="weather-label flex-center gap-1">
            <Droplets size={12} /> Humidity
          </div>
        </div>
        <div className="weather-item">
          <div className="weather-value">{weather.current_precipitation_mm ?? '0.0'} mm</div>
          <div className="weather-label flex-center gap-1">
            <Cloud size={12} /> Current Rain
          </div>
        </div>
        <div className="weather-item">
          <div className="weather-value">{weather.annual_est_rainfall_mm ?? '--'} mm</div>
          <div className="weather-label flex-center gap-1">
            <Cloud size={12} /> Annual Rain (Est)
          </div>
        </div>
        <div className="weather-item">
          <div className="weather-value">{weather.current_wind_speed_kmh ?? '--'} km/h</div>
          <div className="weather-label flex-center gap-1">
            <Wind size={12} /> Wind
          </div>
        </div>
      </div>
      {weather.annual_note && (
        <p className="text-xs text-muted mt-4 font-mono">
          * {weather.annual_note}
        </p>
      )}
    </div>
  )
}
