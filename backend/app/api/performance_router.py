"""Model performance metrics router."""

import json
from pathlib import Path
from fastapi import APIRouter

router = APIRouter()

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"


@router.get("/model-performance")
async def get_model_performance():
    """
    Return actual ML model performance metrics from training reports.
    All numbers are calculated from real training data — no fake metrics.
    """
    result = {}

    for key, fname in [
        ("yield_metrics",        "yield_model_metrics.json"),
        ("recommendation_metrics","recommendation_model_metrics.json"),
        ("secondary_validation", "secondary_validation_metrics.json"),
        ("dataset_analysis",     "dataset_analysis_report.json"),
    ]:
        p = REPORTS_DIR / fname
        if p.exists():
            with open(p) as f:
                result[key] = json.load(f)
        else:
            result[key] = None

    result["models_trained"] = (
        result["yield_metrics"] is not None
        and result["recommendation_metrics"] is not None
    )

    if not result["models_trained"]:
        result["message"] = (
            "Models not yet trained. "
            "Run ml/training/train_yield_model.py and ml/training/train_recommendation_model.py first."
        )

    return result
