"""
Yield Model Training Script
- Time-aware split (1997-2014 train / 2015-2017 val / 2018-2020 test)
- Baseline: Linear Regression
- Model A: Random Forest Regressor
- Model B: XGBoost Regressor
- Auto-selects best model by RMSE on validation set
- Saves best model + preprocessor

Run from project root:
    python ml/training/train_yield_model.py
"""

import json
import logging
import pickle
import sys
import warnings
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.preprocessing.yield_preprocessor import YieldPreprocessor

SAVED_MODELS_DIR = PROJECT_ROOT / "ml" / "saved_models"
REPORTS_DIR = PROJECT_ROOT / "reports"
SAVED_MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, split: str) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # MAPE — skip near-zero targets
    mask = np.abs(y_true) > 0.01
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else None

    metrics = {
        "split": split,
        "n_samples": int(len(y_true)),
        "MAE": float(round(mae, 4)),
        "RMSE": float(round(rmse, 4)),
        "R2": float(round(r2, 4)),
        "MAPE_pct": float(round(mape, 2)) if mape is not None else None,
    }
    logger.info(f"  [{split}] MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}" +
                (f", MAPE={mape:.2f}%" if mape else ""))
    return metrics


def get_feature_importance(model, feature_names: list[str]) -> dict:
    """Extract feature importance from tree models."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        return dict(zip(feature_names, [float(round(v, 6)) for v in importances]))
    return {}


def plot_feature_importance(importances: dict, model_name: str, save_path: Path):
    if not importances:
        return
    items = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:12]
    names, values = zip(*items)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(list(reversed(names)), list(reversed(values)), color="#2ecc71")
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"{model_name} — Feature Importance")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Feature importance plot saved: {save_path}")


def plot_actual_vs_predicted(y_true, y_pred, model_name: str, split: str, save_path: Path):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.3, s=10, color="#3498db")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Yield (tonnes/ha)")
    ax.set_ylabel("Predicted Yield (tonnes/ha)")
    ax.set_title(f"{model_name} — Actual vs Predicted ({split})")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Actual vs predicted plot saved: {save_path}")


def train_and_evaluate(name, model, X_train, y_train, X_val, y_val, X_test, y_test) -> dict:
    logger.info(f"\n{'='*50}")
    logger.info(f"Training: {name}")
    model.fit(X_train, y_train)

    train_metrics = compute_metrics(y_train, model.predict(X_train), "train")
    val_metrics = compute_metrics(y_val, model.predict(X_val), "validation")
    test_metrics = compute_metrics(y_test, model.predict(X_test), "test")

    return {
        "name": name,
        "train": train_metrics,
        "validation": val_metrics,
        "test": test_metrics,
        "model_object": model,
    }


def main():
    logger.info("=" * 60)
    logger.info("YIELD MODEL TRAINING PIPELINE")
    logger.info("=" * 60)

    # -------------------------------------------------------------------------
    # Step 1: Load & preprocess data
    # -------------------------------------------------------------------------
    preprocessor = YieldPreprocessor()
    raw_df = preprocessor.load_raw_data()
    raw_df = preprocessor.clean(raw_df)

    logger.info(f"\nDataset after cleaning: {raw_df.shape}")
    logger.info(f"Year range: {raw_df['year'].min()}–{raw_df['year'].max()}")

    # -------------------------------------------------------------------------
    # Step 2: Time-aware split
    # -------------------------------------------------------------------------
    train_df = raw_df[raw_df["year"] <= 2014]
    val_df   = raw_df[(raw_df["year"] >= 2015) & (raw_df["year"] <= 2017)]
    test_df  = raw_df[raw_df["year"] >= 2018]

    logger.info(f"\nTime-aware split:")
    logger.info(f"  Train (1997–2014): {len(train_df)} rows")
    logger.info(f"  Val   (2015–2017): {len(val_df)} rows")
    logger.info(f"  Test  (2018–2020): {len(test_df)} rows")

    if len(val_df) == 0 or len(test_df) == 0:
        logger.warning("Val or Test set is empty — falling back to 70/15/15 split")
        from sklearn.model_selection import train_test_split
        train_df, temp_df = train_test_split(raw_df, test_size=0.30, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)

    # Fit preprocessor on training data only
    X_train, y_train = preprocessor.fit_transform(train_df)
    X_val  = preprocessor.transform(val_df)
    y_val  = val_df[preprocessor.TARGET_COL].values
    X_test = preprocessor.transform(test_df)
    y_test = test_df[preprocessor.TARGET_COL].values

    logger.info(f"\nFeature matrix shapes:")
    logger.info(f"  X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

    # -------------------------------------------------------------------------
    # Step 3: Define models
    # -------------------------------------------------------------------------
    models = {
        "Linear Regression (Baseline)": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            n_jobs=-1,
            random_state=42,
            verbosity=0,
        ),
    }

    # -------------------------------------------------------------------------
    # Step 4: Train and evaluate
    # -------------------------------------------------------------------------
    results = {}
    for name, model in models.items():
        r = train_and_evaluate(name, model, X_train, y_train, X_val, y_val, X_test, y_test)
        results[name] = r

    # -------------------------------------------------------------------------
    # Step 5: Select best model by validation RMSE
    # -------------------------------------------------------------------------
    best_name = min(results, key=lambda k: results[k]["validation"]["RMSE"])
    best_result = results[best_name]
    best_model = best_result["model_object"]

    logger.info(f"\n{'='*60}")
    logger.info(f"BEST MODEL: {best_name}")
    logger.info(f"  Validation RMSE : {best_result['validation']['RMSE']}")
    logger.info(f"  Validation R²   : {best_result['validation']['R2']}")
    logger.info(f"  Test RMSE       : {best_result['test']['RMSE']}")
    logger.info(f"  Test R²         : {best_result['test']['R2']}")

    # -------------------------------------------------------------------------
    # Step 6: Feature importance
    # -------------------------------------------------------------------------
    feature_names = preprocessor.feature_cols
    feat_imp = get_feature_importance(best_model, feature_names)

    # -------------------------------------------------------------------------
    # Step 7: Save model + preprocessor
    # -------------------------------------------------------------------------
    model_path = SAVED_MODELS_DIR / "yield_model.pkl"
    prep_path  = SAVED_MODELS_DIR / "yield_preprocessor.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    preprocessor.save(prep_path)

    logger.info(f"\nSaved yield model → {model_path}")
    logger.info(f"Saved preprocessor → {prep_path}")

    # -------------------------------------------------------------------------
    # Step 8: Generate plots
    # -------------------------------------------------------------------------
    plot_feature_importance(
        feat_imp, best_name, REPORTS_DIR / "yield_feature_importance.png"
    )
    plot_actual_vs_predicted(
        y_test, best_model.predict(X_test), best_name,
        "Test", REPORTS_DIR / "yield_actual_vs_predicted.png"
    )

    # -------------------------------------------------------------------------
    # Step 9: Save metrics report
    # -------------------------------------------------------------------------
    metrics_report = {
        "best_model": best_name,
        "feature_names": feature_names,
        "feature_importance": feat_imp,
        "split_sizes": {
            "train": int(len(X_train)),
            "validation": int(len(X_val)),
            "test": int(len(X_test)),
        },
        "models": {
            name: {
                "train": r["train"],
                "validation": r["validation"],
                "test": r["test"],
            }
            for name, r in results.items()
        },
    }

    metrics_path = REPORTS_DIR / "yield_model_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_report, f, indent=2)
    logger.info(f"Metrics report saved → {metrics_path}")

    logger.info("\n[+] Yield model training complete!")
    return metrics_report


if __name__ == "__main__":
    main()
