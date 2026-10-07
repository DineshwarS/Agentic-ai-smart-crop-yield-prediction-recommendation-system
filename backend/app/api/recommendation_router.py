"""Crop recommendation API router."""

from datetime import datetime, timezone
import logging

from fastapi import APIRouter, HTTPException

from ..database.mongodb import insert_prediction
from ..schemas.schemas import CropRecommendationRequest, CropRecommendationResponse
from ..services.ml_service import ml_service
from ..services.weather_service import get_weather_for_location

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/recommend-crop", response_model=CropRecommendationResponse)
async def recommend_crop(req: CropRecommendationRequest):
    """
    Recommend the best crop for given soil/weather conditions.
    Automatically fetches live weather if lat/lon provided and weather missing.
    """
    if ml_service.recommendation_model is None:
        raise HTTPException(
            status_code=503,
            detail="Recommendation model not loaded. Run train_recommendation_model.py first.",
        )

    input_data = req.model_dump()
    weather_source = "user_provided"

    # Fetch live weather if needed
    needs_weather = req.temperature is None or req.humidity is None or req.rainfall is None
    if needs_weather and req.latitude is not None and req.longitude is not None:
        try:
            weather = await get_weather_for_location(req.latitude, req.longitude)
            if req.temperature is None:
                input_data["temperature"] = weather.get("current_temperature_c")
            if req.humidity is None:
                input_data["humidity"] = weather.get("current_humidity_pct")
            if req.rainfall is None:
                # Use annual total for rainfall feature (matches training data)
                input_data["rainfall"] = weather.get("annual_est_rainfall_mm")
            weather_source = "live"
        except Exception as e:
            logger.warning(f"Live weather failed: {e}")
            weather_source = "unavailable"

    # Run recommendation
    try:
        result = ml_service.recommend_crop(input_data)
    except Exception as e:
        logger.error(f"Recommendation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")

    # Generate explanation
    top = result["top_crops"][:3]
    explanation = (
        f"Based on your soil conditions (N={req.N}, P={req.P}, K={req.K}, pH={req.ph}) "
        f"and current weather, the model recommends **{result['recommended_crop']}** "
        f"as the most suitable crop. "
    )
    if len(top) > 1:
        others = ", ".join([f"{c['crop']} ({c['score']*100:.1f}%)" for c in top[1:3]])
        explanation += f"Other suitable options include: {others}."

    ts = datetime.now(timezone.utc)

    # Persist to MongoDB
    record = {
        "type": "recommendation",
        "recommended_crop": result["recommended_crop"],
        "top_crops": result["top_crops"],
        "model_used": result["model_used"],
        "weather_source": weather_source,
        "input_data": {k: v for k, v in input_data.items() if v is not None},
        "timestamp": ts,
    }
    prediction_id = await insert_prediction(record)

    return CropRecommendationResponse(
        prediction_id=prediction_id,
        recommended_crop=result["recommended_crop"],
        top_crops=result["top_crops"],
        model_used=result["model_used"],
        input_conditions={
            k: v for k, v in input_data.items()
            if k not in ["latitude", "longitude"] and v is not None
        },
        explanation=explanation,
        weather_source=weather_source,
        timestamp=ts,
    )


@router.get("/recommendable-crops")
async def get_recommendable_crops():
    """Return list of crops in the recommendation model."""
    return {"crops": ml_service.get_recommendable_crops()}
