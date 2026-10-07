# REST API Documentation

The FastAPI backend runs by default at `http://localhost:8000`. Interactive OpenAPI documentation is accessible at `http://localhost:8000/docs`.

---

## 1. Endpoints Overview

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Health check & model load status |
| `POST` | `/api/predict-yield` | Forecast expected crop yield in tonnes/ha |
| `POST` | `/api/recommend-crop` | Recommend top crops based on soil & weather |
| `GET` | `/api/weather` | Retrieve live Open-Meteo weather & annual aggregates |
| `POST` | `/api/agent/chat` | Conversational interface with Gemini Agent |
| `GET` | `/api/predictions` | Query MongoDB prediction history |
| `GET` | `/api/predictions/{id}` | Query single prediction record |
| `GET` | `/api/model-performance` | Retrieve real experimental benchmark metrics |
| `GET` | `/api/crops` | List supported crops in yield engine |
| `GET` | `/api/states` | List supported Indian states |
| `GET` | `/api/seasons` | List supported seasons |
| `GET` | `/api/state-soil/{state}` | Retrieve baseline soil defaults for a state |

---

## 2. Sample Request & Response Payloads

### 2.1 Predict Yield
`POST /api/predict-yield`

```json
{
  "crop": "Rice",
  "season": "Kharif",
  "state": "Punjab",
  "area": 10.0,
  "fertilizer": 1500.0,
  "pesticide": 25.0,
  "year": 2024,
  "N": 150.0,
  "P": 50.0,
  "K": 40.0,
  "pH": 8.0,
  "avg_temp_c": 25.0,
  "total_rainfall_mm": 800.0,
  "avg_humidity_percent": 65.0
}
```

**Response (200 OK)**:
```json
{
  "prediction_id": "66d1f...",
  "crop": "Rice",
  "state": "Punjab",
  "season": "Kharif",
  "predicted_yield_tonnes_ha": 3.8241,
  "model_used": "XGBoost",
  "weather_source": "user_provided",
  "feature_importance": {
    "crop": 0.4521,
    "state": 0.1832,
    "area": 0.1245
  },
  "input_summary": { ... },
  "disclaimer": "Prediction based on available historical regional agricultural data...",
  "timestamp": "2026-08-30T15:20:00Z"
}
```
