"""Yield prediction API router."""

from datetime import datetime, timezone
import logging

from fastapi import APIRouter, HTTPException

from ..database.mongodb import insert_prediction
from ..schemas.schemas import YieldPredictionRequest, YieldPredictionResponse
from ..services.ml_service import ml_service
from ..services.weather_service import get_weather_for_location

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/predict-yield", response_model=YieldPredictionResponse)
async def predict_yield(req: YieldPredictionRequest):
    """
    Predict crop yield in tonnes/hectare.
    Automatically fetches live weather if latitude/longitude is provided.
    Soil values are auto-filled from state lookup if not provided.
    """
    if ml_service.yield_model is None:
        raise HTTPException(
            status_code=503,
            detail="Yield model not loaded. Run train_yield_model.py first.",
        )

    # Build feature dict
    input_data = req.model_dump()
    weather_source = "user_provided"

    # Fetch live weather if coordinates given and weather not fully provided
    needs_weather = any(
        req.model_dump().get(f) is None
        for f in ["avg_temp_c", "total_rainfall_mm", "avg_humidity_percent"]
    )

    if needs_weather and req.latitude is not None and req.longitude is not None:
        try:
            weather = await get_weather_for_location(req.latitude, req.longitude)
            input_data["avg_temp_c"] = weather.get("annual_est_temp_c")
            input_data["total_rainfall_mm"] = weather.get("annual_est_rainfall_mm")
            input_data["avg_humidity_percent"] = weather.get("annual_est_humidity_pct")
            weather_source = "live"
        except Exception as e:
            logger.warning(f"Live weather failed: {e}. Proceeding without live weather.")
            weather_source = "unavailable"

    # Run prediction
    try:
        result = ml_service.predict_yield(input_data)
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    ts = datetime.now(timezone.utc)

    # Save to MongoDB
    record = {
        "type": "yield",
        "crop": req.crop,
        "state": req.state,
        "season": req.season,
        "predicted_yield_tonnes_ha": result["predicted_yield_tonnes_ha"],
        "model_used": result["model_used"],
        "weather_source": weather_source,
        "input_data": {
            k: v for k, v in input_data.items()
            if k not in ["latitude", "longitude"] and v is not None
        },
        "timestamp": ts,
    }
    prediction_id = await insert_prediction(record)

    return YieldPredictionResponse(
        prediction_id=prediction_id,
        crop=req.crop,
        state=req.state,
        season=req.season,
        predicted_yield_tonnes_ha=result["predicted_yield_tonnes_ha"],
        model_used=result["model_used"],
        weather_source=weather_source,
        feature_importance=result.get("feature_importance", {}),
        input_summary={
            k: v for k, v in input_data.items()
            if k not in ["latitude", "longitude"] and v is not None
        },
        timestamp=ts,
    )


@router.get("/crops")
async def get_crops():
    """Return list of crops supported by the yield model."""
    return {"crops": ml_service.get_known_crops()}


@router.get("/states")
async def get_states():
    """Return list of states supported by the models."""
    return {"states": ml_service.get_known_states()}


@router.get("/seasons")
async def get_seasons():
    """Return list of seasons supported by the yield model."""
    return {"seasons": ml_service.get_known_seasons()}


@router.get("/state-soil/{state}")
async def get_state_soil(state: str):
    """Return default soil values for a given state."""
    soil = ml_service.get_soil_for_state(state)
    if not soil:
        raise HTTPException(status_code=404, detail=f"State '{state}' not found in soil database.")
    return soil
