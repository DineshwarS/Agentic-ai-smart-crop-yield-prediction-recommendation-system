from fastapi import APIRouter, HTTPException, Query
from ..services.weather_service import get_weather_for_location

router = APIRouter()


@router.get("/weather")
async def get_weather(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
    location: str = Query("", description="Location name (display only)"),
):
    """
    Fetch live weather from Open-Meteo for the given coordinates.
    Returns current conditions + annual estimates for ML model compatibility.
    """
    if not (-90 <= lat <= 90):
        raise HTTPException(status_code=400, detail="Latitude must be between -90 and 90.")
    if not (-180 <= lon <= 180):
        raise HTTPException(status_code=400, detail="Longitude must be between -180 and 180.")

    try:
        weather = await get_weather_for_location(lat, lon)
        weather["location"] = location or f"{lat:.4f}, {lon:.4f}"
        return weather
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
