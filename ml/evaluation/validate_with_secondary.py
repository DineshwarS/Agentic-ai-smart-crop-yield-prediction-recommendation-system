"""
Secondary Dataset Validation
Uses the saved yield model to evaluate on the external validation dataset.
Units: Validation dataset has Yield_kg_per_ha → convert to tonnes/ha (÷ 1000)
"""

import json
import logging
import pickle
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.preprocessing.yield_preprocessor import YieldPreprocessor

VALIDATE_DIR = PROJECT_ROOT / "Validate test model dataset"
SAVED_MODELS_DIR = PROJECT_ROOT / "ml" / "saved_models"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def main():
    logger.info("=" * 60)
    logger.info("SECONDARY DATASET VALIDATION")
    logger.info("=" * 60)

    # Load saved model and preprocessor
    model_path = SAVED_MODELS_DIR / "yield_model.pkl"
    prep_path  = SAVED_MODELS_DIR / "yield_preprocessor.pkl"

    if not model_path.exists() or not prep_path.exists():
        logger.error("Saved model not found. Run train_yield_model.py first.")
        return

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    preprocessor: YieldPreprocessor = YieldPreprocessor.load(prep_path)

    # Load validation dataset
    val_path = VALIDATE_DIR / "Crops yield and Historical Dataset.csv"
    logger.info(f"Loading {val_path} ...")
    val_df = pd.read_csv(val_path)
    val_df.columns = val_df.columns.str.strip()
    logger.info(f"Validation dataset: {val_df.shape}")

    # -------------------------------------------------------------------------
    # Unit conversion: kg/ha → tonnes/ha
    # -------------------------------------------------------------------------
    val_df["yield_tonnes_ha"] = val_df["Yield_kg_per_ha"] / 1000.0
    logger.info("Converted Yield_kg_per_ha → yield_tonnes_ha (÷ 1000)")

    # -------------------------------------------------------------------------
    # Map validation columns to training feature names
    # -------------------------------------------------------------------------
    # Training features: crop, season, state, area, fertilizer, pesticide,
    #                    year, N, P, K, pH, avg_temp_c, total_rainfall_mm, avg_humidity_percent
    # Validation columns: Dist Code, Year, State Name, Dist Name, Crop, Area_ha,
    #                     N_req_kg_per_ha, P_req_kg_per_ha, K_req_kg_per_ha,
    #                     Temperature_C, Humidity_%, pH, Rainfall_mm, ...

    # Normalize crop and state names for matching
    crop_map = {c.lower(): c for c in preprocessor.get_class_names("crop")}
    state_map = {s.lower(): s for s in preprocessor.get_class_names("state")}

    val_crops = val_df["Crop"].astype(str).str.strip().str.lower()
    val_states = val_df["State Name"].astype(str).str.strip().str.lower()

    # Filter to compatible rows
    mask = val_crops.isin(crop_map.keys()) & val_states.isin(state_map.keys())
    val_filtered = val_df[mask].copy()

    logger.info(f"\nFiltered validation rows with matching crop & state: {len(val_filtered)}")
    if len(val_filtered) < 10:
        logger.warning("Too few compatible rows — skipping external validation.")
        return

    compat = pd.DataFrame()
    compat["crop"]               = val_filtered["Crop"].astype(str).str.strip().str.lower().map(crop_map)
    compat["season"]             = "Whole Year"   # default season
    compat["state"]              = val_filtered["State Name"].astype(str).str.strip().str.lower().map(state_map)
    compat["area"]               = val_filtered["Area_ha"].values
    compat["fertilizer"]         = (val_filtered["Total_N_kg"] + val_filtered["Total_P_kg"] + val_filtered["Total_K_kg"]).values
    compat["pesticide"]          = 0.0
    compat["year"]               = val_filtered["Year"].values
    compat["N"]                  = val_filtered["N_req_kg_per_ha"].values
    compat["P"]                  = val_filtered["P_req_kg_per_ha"].values
    compat["K"]                  = val_filtered["K_req_kg_per_ha"].values
    compat["pH"]                 = val_filtered["pH"].values
    compat["avg_temp_c"]         = val_filtered["Temperature_C"].values
    compat["total_rainfall_mm"]  = val_filtered["Rainfall_mm"].values
    compat["avg_humidity_percent"] = val_filtered["Humidity_%"].values
    compat["yield"]              = val_filtered["yield_tonnes_ha"].values

    compat = compat.dropna(subset=["yield"])
    compat = compat[compat["yield"] > 0]
    logger.info(f"Compatible rows after filtering: {len(compat)}")

    # -------------------------------------------------------------------------
    # Predict and evaluate
    # -------------------------------------------------------------------------
    X_val = preprocessor.transform(compat)
    y_true = compat["yield"].values
    y_pred = model.predict(X_val)

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    mask = np.abs(y_true) > 0.01
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else None

    logger.info(f"\n{'='*60}")
    logger.info("EXTERNAL VALIDATION RESULTS")
    logger.info(f"  Samples evaluated : {len(y_true)}")
    logger.info(f"  MAE               : {mae:.4f} tonnes/ha")
    logger.info(f"  RMSE              : {rmse:.4f} tonnes/ha")
    logger.info(f"  R²                : {r2:.4f}")
    if mape:
        logger.info(f"  MAPE              : {mape:.2f}%")

    report = {
        "dataset": "Crops yield and Historical Dataset.csv",
        "unit_conversion": "Yield_kg_per_ha / 1000 = tonnes/ha",
        "compatible_rows": int(len(y_true)),
        "notes": [
            "season set to 'Whole Year' (not present in validation dataset)",
            "pesticide set to 0 (not available in validation dataset)",
            "Only crops and states present in training encoder are evaluated",
        ],
        "metrics": {
            "MAE": float(round(mae, 4)),
            "RMSE": float(round(rmse, 4)),
            "R2": float(round(r2, 4)),
            "MAPE_pct": float(round(mape, 2)) if mape else None,
        },
    }

    out_path = REPORTS_DIR / "secondary_validation_metrics.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"\nValidation report saved -> {out_path}")
    logger.info("[+] Secondary validation complete!")


if __name__ == "__main__":
    main()
