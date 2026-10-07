from fastapi import APIRouter, HTTPException, Query
from ..database.mongodb import get_predictions, get_prediction_by_id

router = APIRouter()


@router.get("/predictions")
async def list_predictions(limit: int = Query(50, le=200)):
    """Fetch recent predictions from MongoDB."""
    records = await get_predictions(limit=limit)
    return {"predictions": records, "count": len(records)}


@router.get("/predictions/{prediction_id}")
async def get_prediction(prediction_id: str):
    """Fetch a single prediction by MongoDB ID."""
    record = await get_prediction_by_id(prediction_id)
    if not record:
        raise HTTPException(status_code=404, detail="Prediction not found.")
    return record
