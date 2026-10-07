"""
Weather Service — Open-Meteo API integration.
Fetches live weather and aggregates to annual/seasonal estimates
compatible with the training dataset feature definitions:
  - avg_temp_c         = annual average temperature (°C)
  - total_rainfall_mm  = annual total precipitation (mm)
  - avg_humidity_percent = annual average relative humidity (%)

Open-Meteo is free, no API key required.
Docs: https://open-meteo.com/en/docs
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


async def get_live_weather(lat: float, lon: float) -> dict:
    """
    Fetch current weather + 7-day forecast from Open-Meteo.
    Also estimates annual aggregates for ML model compatibility.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
            "weather_code",
        ],
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
        ],
        "forecast_days": 7,
        "timezone": "Asia/Kolkata",
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(OPEN_METEO_URL, params=params)
        response.raise_for_status()
        data = response.json()

    current = data.get("current", {})
    hourly = data.get("hourly", {})

    # Current readings
    current_temp = current.get("temperature_2m", None)
    current_humidity = current.get("relative_humidity_2m", None)
    current_precip = current.get("precipitation", None)
    current_wind = current.get("wind_speed_10m", None)

    # 7-day aggregates as proxy (clearly labeled as estimates)
    temps = [t for t in hourly.get("temperature_2m", []) if t is not None]
    humidities = [h for h in hourly.get("relative_humidity_2m", []) if h is not None]
    precips = [p for p in hourly.get("precipitation", []) if p is not None]

    # For model inference, we need annual-scale estimates
    # We use historical archive for the past 365 days
    annual_estimates = await _get_annual_estimates(lat, lon)

    return {
        "latitude": lat,
        "longitude": lon,
        "current_temperature_c": round(current_temp, 2) if current_temp is not None else None,
        "current_humidity_pct": round(current_humidity, 2) if current_humidity is not None else None,
        "current_precipitation_mm": round(current_precip, 2) if current_precip is not None else None,
        "current_wind_speed_kmh": round(current_wind, 2) if current_wind is not None else None,
        # 7-day averages (for dashboard display)
        "week_avg_temp_c": round(sum(temps) / len(temps), 2) if temps else None,
        "week_avg_humidity_pct": round(sum(humidities) / len(humidities), 2) if humidities else None,
        "week_total_precip_mm": round(sum(precips), 2) if precips else None,
        # Annual estimates for ML model compatibility
        "annual_est_temp_c": annual_estimates.get("avg_temp_c"),
        "annual_est_rainfall_mm": annual_estimates.get("total_rainfall_mm"),
        "annual_est_humidity_pct": annual_estimates.get("avg_humidity_pct"),
        "annual_note": annual_estimates.get("note", ""),
        "data_source": "Open-Meteo API",
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
    }


async def _get_annual_estimates(lat: float, lon: float) -> dict:
    """
    Fetch past 365 days of daily data from Open-Meteo archive API.
    Computes annual averages/totals matching training feature semantics.
    """
    end_date = datetime.now(timezone.utc).date() - timedelta(days=1)
    start_date = end_date - timedelta(days=364)

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": [
            "temperature_2m_mean",
            "precipitation_sum",
            "relative_humidity_2m_mean",
        ],
        "timezone": "Asia/Kolkata",
    }

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(OPEN_METEO_ARCHIVE_URL, params=params)
            response.raise_for_status()
            data = response.json()

        daily = data.get("daily", {})
        temps = [t for t in daily.get("temperature_2m_mean", []) if t is not None]
        precips = [p for p in daily.get("precipitation_sum", []) if p is not None]
        humidities = [h for h in daily.get("relative_humidity_2m_mean", []) if h is not None]

        avg_temp = round(sum(temps) / len(temps), 2) if temps else None
        total_rain = round(sum(precips), 2) if precips else None
        avg_hum = round(sum(humidities) / len(humidities), 2) if humidities else None

        return {
            "avg_temp_c": avg_temp,
            "total_rainfall_mm": total_rain,
            "avg_humidity_pct": avg_hum,
            "note": (
                f"Annual estimates from {start_date} to {end_date} "
                f"(past 365 days via Open-Meteo Archive API)."
            ),
        }
    except Exception as e:
        logger.warning(f"Could not fetch annual estimates: {e}")
        return {
            "avg_temp_c": None,
            "total_rainfall_mm": None,
            "avg_humidity_pct": None,
            "note": "Annual estimates unavailable — archive API error.",
        }


async def get_weather_for_location(lat: float, lon: float) -> dict:
    """Public entry point used by routers and agent tools."""
    try:
        return await get_live_weather(lat, lon)
    except httpx.HTTPStatusError as e:
        logger.error(f"Open-Meteo HTTP error: {e}")
        raise RuntimeError(f"Weather API error: {e.response.status_code}")
    except httpx.RequestError as e:
        logger.error(f"Open-Meteo connection error: {e}")
        raise RuntimeError("Weather API unavailable. Check network connection.")
