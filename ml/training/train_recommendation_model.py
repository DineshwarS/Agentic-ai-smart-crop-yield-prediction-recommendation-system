"""
Crop Recommendation Model Training Script
- Stratified 80/20 split
- Random Forest Classifier vs XGBoost Classifier
- Evaluates: Accuracy, Precision, Recall, F1, Confusion Matrix
- Auto-selects best model by F1-macro

Run from project root:
    python ml/training/train_recommendation_model.py
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
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.preprocessing.recommendation_preprocessor import RecommendationPreprocessor

DATASETS_DIR = PROJECT_ROOT
SAVED_MODELS_DIR = PROJECT_ROOT / "ml" / "saved_models"
REPORTS_DIR = PROJECT_ROOT / "reports"
SAVED_MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


def compute_class_metrics(y_true, y_pred, classes: list[str]) -> dict:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)

    logger.info(f"  Accuracy : {acc:.4f}")
    logger.info(f"  Precision: {prec:.4f} (macro)")
    logger.info(f"  Recall   : {rec:.4f} (macro)")
    logger.info(f"  F1       : {f1:.4f} (macro)")
    return {
        "accuracy": float(round(acc, 4)),
        "precision_macro": float(round(prec, 4)),
        "recall_macro": float(round(rec, 4)),
        "f1_macro": float(round(f1, 4)),
        "per_class": report,
    }


def plot_confusion_matrix(y_true, y_pred, classes, model_name: str, save_path: Path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(12, len(classes)), max(10, len(classes) - 2)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax, linewidths=0.5)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{model_name} — Confusion Matrix")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    logger.info(f"Confusion matrix saved: {save_path}")


def main():
    logger.info("=" * 60)
    logger.info("CROP RECOMMENDATION MODEL TRAINING")
    logger.info("=" * 60)

    # -------------------------------------------------------------------------
    # Step 1: Load and preprocess
    # -------------------------------------------------------------------------
    preprocessor = RecommendationPreprocessor()
    raw_df = preprocessor.load_data(DATASETS_DIR / "Crop_recommendation.csv")

    X, y = preprocessor.fit_transform(raw_df)
    classes = preprocessor.get_classes()
    logger.info(f"\nClasses ({len(classes)}): {classes}")

    # -------------------------------------------------------------------------
    # Step 2: Stratified train/test split
    # -------------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    logger.info(f"\nSplit: {len(X_train)} train / {len(X_test)} test")

    # -------------------------------------------------------------------------
    # Step 3: Define models
    # -------------------------------------------------------------------------
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            n_jobs=-1,
            random_state=42,
            verbosity=0,
        ),
    }

    # -------------------------------------------------------------------------
    # Step 4: Train, evaluate, CV
    # -------------------------------------------------------------------------
    results = {}
    for name, model in models.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Training: {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        test_metrics = compute_class_metrics(y_pred, y_test, classes)

        # 5-fold cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
        logger.info(f"  CV F1 (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        results[name] = {
            "model_object": model,
            "test_metrics": test_metrics,
            "cv_f1_mean": float(round(cv_scores.mean(), 4)),
            "cv_f1_std":  float(round(cv_scores.std(), 4)),
            "y_pred": y_pred,
        }

    # -------------------------------------------------------------------------
    # Step 5: Select best model by test F1-macro
    # -------------------------------------------------------------------------
    best_name = max(results, key=lambda k: results[k]["test_metrics"]["f1_macro"])
    best_result = results[best_name]
    best_model = best_result["model_object"]

    logger.info(f"\n{'='*60}")
    logger.info(f"BEST MODEL: {best_name}")
    logger.info(f"  Test Accuracy  : {best_result['test_metrics']['accuracy']}")
    logger.info(f"  Test F1 (macro): {best_result['test_metrics']['f1_macro']}")
    logger.info(f"  CV F1 (macro)  : {best_result['cv_f1_mean']} ± {best_result['cv_f1_std']}")

    # -------------------------------------------------------------------------
    # Step 6: Feature importance (for recommendation model)
    # -------------------------------------------------------------------------
    feat_imp = {}
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        feat_imp = dict(zip(preprocessor.FEATURE_COLS, [float(round(v, 6)) for v in importances]))

    # -------------------------------------------------------------------------
    # Step 7: Confusion matrix plot
    # -------------------------------------------------------------------------
    plot_confusion_matrix(
        y_test, best_result["y_pred"], classes, best_name,
        REPORTS_DIR / "recommendation_confusion_matrix.png"
    )

    # -------------------------------------------------------------------------
    # Step 8: Save model + preprocessor
    # -------------------------------------------------------------------------
    model_path = SAVED_MODELS_DIR / "recommendation_model.pkl"
    prep_path  = SAVED_MODELS_DIR / "recommendation_preprocessor.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    preprocessor.save(prep_path)

    logger.info(f"\nSaved recommendation model → {model_path}")
    logger.info(f"Saved preprocessor → {prep_path}")

    # -------------------------------------------------------------------------
    # Step 9: Save metrics report
    # -------------------------------------------------------------------------
    metrics_report = {
        "best_model": best_name,
        "feature_names": preprocessor.FEATURE_COLS,
        "classes": classes,
        "feature_importance": feat_imp,
        "split_sizes": {
            "train": int(len(X_train)),
            "test": int(len(X_test)),
        },
        "models": {
            name: {
                "test_metrics": r["test_metrics"],
                "cv_f1_mean": r["cv_f1_mean"],
                "cv_f1_std":  r["cv_f1_std"],
            }
            for name, r in results.items()
        },
    }

    metrics_path = REPORTS_DIR / "recommendation_model_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_report, f, indent=2)
    logger.info(f"Metrics saved → {metrics_path}")

    logger.info("\n[+] Recommendation model training complete!")
    return metrics_report


if __name__ == "__main__":
    main()
