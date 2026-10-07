"""Pydantic request/response schemas."""

from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class BaseAppModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())


# ─────────────────── Yield Prediction ────────────────────

class YieldPredictionRequest(BaseAppModel):
    crop: str = Field(..., examples=["Rice"])
    season: str = Field(..., examples=["Kharif"])
    state: str = Field(..., examples=["Punjab"])
    area: float = Field(..., gt=0, examples=[100.0], description="Area in hectares")
    fertilizer: float = Field(..., ge=0, examples=[50000.0], description="Total fertilizer in kg")
    pesticide: float = Field(..., ge=0, examples=[500.0], description="Pesticide in kg")
    year: int = Field(..., ge=1997, le=2030, examples=[2024])
    # Soil params (optional — pulled from state lookup if not provided)
    N: Optional[float] = Field(None, ge=0, examples=[85.0])
    P: Optional[float] = Field(None, ge=0, examples=[40.0])
    K: Optional[float] = Field(None, ge=0, examples=[35.0])
    pH: Optional[float] = Field(None, ge=0, le=14, examples=[7.5])
    # Weather (optional — fetched live if not provided)
    avg_temp_c: Optional[float] = Field(None, examples=[28.0])
    total_rainfall_mm: Optional[float] = Field(None, examples=[1200.0])
    avg_humidity_percent: Optional[float] = Field(None, examples=[70.0])
    # Location for live weather
    latitude: Optional[float] = Field(None, examples=[30.9])
    longitude: Optional[float] = Field(None, examples=[75.85])


class YieldPredictionResponse(BaseAppModel):
    prediction_id: Optional[str] = None
    crop: str
    state: str
    season: str
    predicted_yield_tonnes_ha: float
    model_used: str
    weather_source: str   # "live" | "historical" | "user_provided"
    feature_importance: dict
    input_summary: dict
    disclaimer: str = (
        "Prediction based on available historical regional agricultural data and "
        "current/estimated weather conditions. Not a guarantee of actual farm yield."
    )
    timestamp: datetime


# ─────────────────── Crop Recommendation ─────────────────

class CropRecommendationRequest(BaseAppModel):
    N: float = Field(..., ge=0, examples=[85.0])
    P: float = Field(..., ge=0, examples=[40.0])
    K: float = Field(..., ge=0, examples=[35.0])
    ph: float = Field(..., ge=0, le=14, examples=[6.5])
    temperature: Optional[float] = Field(None, examples=[28.0])
    humidity: Optional[float] = Field(None, examples=[70.0])
    rainfall: Optional[float] = Field(None, examples=[1200.0])
    latitude: Optional[float] = Field(None, examples=[30.9])
    longitude: Optional[float] = Field(None, examples=[75.85])
    state: Optional[str] = Field(None, examples=["Punjab"])


class CropRecommendationResponse(BaseAppModel):
    prediction_id: Optional[str] = None
    recommended_crop: str
    top_crops: list[dict]   # [{crop, probability/score}]
    model_used: str
    input_conditions: dict
    explanation: str
    weather_source: str
    timestamp: datetime


# ─────────────────── Weather ─────────────────────────────

class WeatherResponse(BaseAppModel):
    location: str
    latitude: float
    longitude: float
    current_temperature_c: float
    current_humidity_pct: float
    current_precipitation_mm: float
    current_wind_speed_kmh: float
    annual_est_temp_c: float
    annual_est_rainfall_mm: float
    annual_est_humidity_pct: float
    data_source: str = "Open-Meteo API"
    retrieved_at: datetime


# ─────────────────── Agent Chat ──────────────────────────

class AgentMessage(BaseAppModel):
    role: str   # "user" | "assistant"
    content: str


class AgentChatRequest(BaseAppModel):
    message: str
    session_id: Optional[str] = None
    context: Optional[dict] = None   # last prediction, location, etc.


class AgentChatResponse(BaseAppModel):
    reply: str
    session_id: str
    tool_calls_made: list[str] = []
    structured_data: Optional[dict] = None


# ─────────────────── Prediction History ──────────────────

class PredictionRecord(BaseAppModel):
    id: Optional[str] = None
    type: str   # "yield" | "recommendation"
    crop: Optional[str] = None
    state: Optional[str] = None
    predicted_yield: Optional[float] = None
    recommended_crop: Optional[str] = None
    model_used: str
    timestamp: datetime
    input_data: dict
