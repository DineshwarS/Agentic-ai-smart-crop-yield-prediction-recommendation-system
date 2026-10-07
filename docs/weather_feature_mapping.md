# Weather Feature Mapping & Aggregation Layer

## 1. Problem Statement

A frequent flaw in naive ML agricultural implementations is directly plugging live hourly or daily weather (e.g., today's rainfall = 5 mm) into a model that was trained on **annual** or **seasonal** aggregated figures (e.g., annual rainfall = 1200 mm). Doing so leads to catastrophic underestimation.

This document formalizes the transformation layer between **Open-Meteo Live APIs** and **Model Feature Expectations**.

---

## 2. Feature Comparison & Mapping

| Model Feature Name | Training Feature Definition | Training Units | Live Open-Meteo Source Endpoint | Transformation / Aggregation Method |
|---|---|---|---|---|
| `avg_temp_c` | Annual average temperature | °C | `archive-api.open-meteo.com/v1/archive` | Mean of daily `temperature_2m_mean` over past 365 days |
| `total_rainfall_mm` | Annual cumulative precipitation | mm | `archive-api.open-meteo.com/v1/archive` | Sum of daily `precipitation_sum` over past 365 days |
| `avg_humidity_percent` | Annual average relative humidity | % | `archive-api.open-meteo.com/v1/archive` | Mean of daily `relative_humidity_2m_mean` over past 365 days |
| `temperature` (Rec Model) | Growing season ambient temperature | °C | `api.open-meteo.com/v1/forecast` | Current `temperature_2m` / 7-day average |
| `humidity` (Rec Model) | Growing season relative humidity | % | `api.open-meteo.com/v1/forecast` | Current `relative_humidity_2m` |
| `rainfall` (Rec Model) | Crop growing cycle rainfall | mm | `api.open-meteo.com/v1/forecast` | Monthly/seasonal estimate from regional baseline or archive aggregation |

---

## 3. Transformation Implementation

The aggregation is implemented in [`backend/app/services/weather_service.py`](file:///backend/app/services/weather_service.py):

```python
async def _get_annual_estimates(lat: float, lon: float) -> dict:
    end_date = datetime.utcnow().date() - timedelta(days=1)
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
    # Computes annual total rainfall and average temperature over 365 daily points
```

---

## 4. Fallback Protocol
If Open-Meteo Archive API is temporarily unavailable or experiences a timeout:
1. The system logs a warning and marks `weather_source = "historical"`.
2. Static state-level historical baseline averages from `state_weather_data_1997_2020.csv` are applied.
3. The farmer is explicitly notified on the UI result panel that an estimate was used.
