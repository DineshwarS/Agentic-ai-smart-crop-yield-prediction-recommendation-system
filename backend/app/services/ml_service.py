"""
ML Service — loads saved models and provides prediction functions.
Models are loaded once at startup and reused for all requests.
"""

import csv
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SAVED_MODELS_DIR = PROJECT_ROOT / "ml" / "saved_models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Soil lookup table — indexed by state
_SOIL_LOOKUP: dict = {}
_STATE_SOIL_PATH = (
    PROJECT_ROOT
    / "Crop Yield Data with Soil and Weather Dataset"
    / "state_soil_data.csv"
)

# Crop Requirements index (N, P, K, pH, climate ranges)
_CROP_REQUIREMENTS: dict = {}
_CROP_REC_CSV_PATH = PROJECT_ROOT / "Crop_recommendation.csv"

# Major Indian Locations geocoding table (State, Lat, Lon)
_LOCATION_COORDINATES: dict = {
    "punjab": ("Punjab", 30.90, 75.85),
    "ludhiana": ("Punjab", 30.90, 75.85),
    "amritsar": ("Punjab", 31.63, 74.87),
    "jalandhar": ("Punjab", 31.32, 75.57),
    "patiala": ("Punjab", 30.33, 76.38),
    "bathinda": ("Punjab", 30.21, 74.94),
    "chandigarh": ("Punjab", 30.73, 76.77),
    "haryana": ("Haryana", 29.05, 76.08),
    "karnal": ("Haryana", 29.68, 76.98),
    "hisar": ("Haryana", 29.14, 75.72),
    "ambala": ("Haryana", 30.37, 76.77),
    "maharashtra": ("Maharashtra", 19.75, 75.71),
    "pune": ("Maharashtra", 18.52, 73.85),
    "nagpur": ("Maharashtra", 21.14, 79.08),
    "nashik": ("Maharashtra", 19.99, 73.78),
    "aurangabad": ("Maharashtra", 19.87, 75.34),
    "tamil nadu": ("Tamil Nadu", 11.12, 78.65),
    "chennai": ("Tamil Nadu", 13.08, 80.27),
    "coimbatore": ("Tamil Nadu", 11.01, 76.95),
    "madurai": ("Tamil Nadu", 9.92, 78.11),
    "thanjavur": ("Tamil Nadu", 10.78, 79.13),
    "uttar pradesh": ("Uttar Pradesh", 26.84, 80.94),
    "lucknow": ("Uttar Pradesh", 26.84, 80.94),
    "varanasi": ("Uttar Pradesh", 25.31, 82.97),
    "kanpur": ("Uttar Pradesh", 26.44, 80.33),
    "agra": ("Uttar Pradesh", 27.17, 78.00),
    "karnataka": ("Karnataka", 15.31, 75.71),
    "bengaluru": ("Karnataka", 12.97, 77.59),
    "mysuru": ("Karnataka", 12.29, 76.63),
    "belagavi": ("Karnataka", 15.84, 74.49),
    "andhra pradesh": ("Andhra Pradesh", 15.91, 79.74),
    "vijayawada": ("Andhra Pradesh", 16.50, 80.64),
    "visakhapatnam": ("Andhra Pradesh", 17.68, 83.21),
    "guntur": ("Andhra Pradesh", 16.30, 80.43),
    "gujarat": ("Gujarat", 22.25, 71.19),
    "ahmedabad": ("Gujarat", 23.02, 72.57),
    "surat": ("Gujarat", 21.17, 72.83),
    "rajkot": ("Gujarat", 22.30, 70.80),
    "madhya pradesh": ("Madhya Pradesh", 22.97, 78.65),
    "bhopal": ("Madhya Pradesh", 23.25, 77.41),
    "indore": ("Madhya Pradesh", 22.71, 75.85),
    "west bengal": ("West Bengal", 22.98, 87.85),
    "kolkata": ("West Bengal", 22.57, 88.36),
    "burdwan": ("West Bengal", 23.23, 87.86),
    "bihar": ("Bihar", 25.09, 85.31),
    "patna": ("Bihar", 25.59, 85.13),
    "muzaffarpur": ("Bihar", 26.12, 85.39),
    "rajasthan": ("Rajasthan", 27.02, 74.21),
    "jaipur": ("Rajasthan", 26.91, 75.78),
    "jodhpur": ("Rajasthan", 26.23, 73.02),
    "kerala": ("Kerala", 10.85, 76.27),
    "kochi": ("Kerala", 9.93, 76.26),
    "thiruvananthapuram": ("Kerala", 8.52, 76.93),
    "telangana": ("Telangana", 18.11, 79.01),
    "hyderabad": ("Telangana", 17.38, 78.48),
    "odisha": ("Odisha", 20.95, 85.09),
    "bhubaneswar": ("Odisha", 20.29, 85.82),
    "assam": ("Assam", 26.20, 92.93),
    "guwahati": ("Assam", 26.14, 91.73),
    "jharkhand": ("Jharkhand", 23.61, 85.27),
    "ranchi": ("Jharkhand", 23.34, 85.30),
    "chhattisgarh": ("Chhattisgarh", 21.27, 81.86),
    "raipur": ("Chhattisgarh", 21.25, 81.62),
    "uttarakhand": ("Uttarakhand", 30.06, 79.01),
    "dehradun": ("Uttarakhand", 30.31, 78.03),
    "himachal pradesh": ("Himachal Pradesh", 31.10, 77.17),
    "shimla": ("Himachal Pradesh", 31.10, 77.17),
}


def _load_soil_lookup():
    global _SOIL_LOOKUP
    if _STATE_SOIL_PATH.exists():
        with open(_STATE_SOIL_PATH, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for raw_row in reader:
                row = {k.strip(): v for k, v in raw_row.items() if k}
                if "state" in row:
                    state = row["state"].strip()
                    _SOIL_LOOKUP[state] = {
                        "N": float(row.get("N", 0)),
                        "P": float(row.get("P", 0)),
                        "K": float(row.get("K", 0)),
                        "pH": float(row.get("pH", 7.0)),
                    }
        logger.info(f"Soil lookup loaded for {len(_SOIL_LOOKUP)} states")


def _load_crop_requirements():
    """Load empirical crop NPK, pH, and climate requirements from Crop_recommendation.csv."""
    global _CROP_REQUIREMENTS
    if _CROP_REC_CSV_PATH.exists():
        try:
            df = pd.read_csv(_CROP_REC_CSV_PATH)
            for crop, g in df.groupby("label"):
                c_name = str(crop).strip().lower()
                _CROP_REQUIREMENTS[c_name] = {
                    "crop": c_name.capitalize(),
                    "soil_pH": {
                        "optimal": round(float(g["ph"].mean()), 2),
                        "range": f"{round(float(g['ph'].min()), 1)} - {round(float(g['ph'].max()), 1)}",
                        "min": round(float(g["ph"].min()), 2),
                        "max": round(float(g["ph"].max()), 2),
                    },
                    "nitrogen_N_kg_ha": {
                        "optimal": round(float(g["N"].mean()), 1),
                        "range": f"{int(g['N'].min())} - {int(g['N'].max())} kg/ha",
                        "min": int(g["N"].min()),
                        "max": int(g["N"].max()),
                    },
                    "phosphorus_P_kg_ha": {
                        "optimal": round(float(g["P"].mean()), 1),
                        "range": f"{int(g['P'].min())} - {int(g['P'].max())} kg/ha",
                        "min": int(g["P"].min()),
                        "max": int(g["P"].max()),
                    },
                    "potassium_K_kg_ha": {
                        "optimal": round(float(g["K"].mean()), 1),
                        "range": f"{int(g['K'].min())} - {int(g['K'].max())} kg/ha",
                        "min": int(g["K"].min()),
                        "max": int(g["K"].max()),
                    },
                    "climate": {
                        "temperature_c": f"{round(float(g['temperature'].min()), 1)} - {round(float(g['temperature'].max()), 1)} °C (optimum ~{round(float(g['temperature'].mean()), 1)} °C)",
                        "humidity_pct": f"{round(float(g['humidity'].min()), 1)}% - {round(float(g['humidity'].max()), 1)}% (optimum ~{round(float(g['humidity'].mean()), 1)}%)",
                        "rainfall_mm": f"{round(float(g['rainfall'].min()), 1)} - {round(float(g['rainfall'].max()), 1)} mm (optimum ~{round(float(g['rainfall'].mean()), 1)} mm)",
                    },
                    "agronomic_advice": _get_agronomic_advice(c_name),
                }
            logger.info(f"Loaded crop requirements for {len(_CROP_REQUIREMENTS)} crops")
        except Exception as e:
            logger.warning(f"Could not parse Crop_recommendation.csv: {e}")


def _get_agronomic_advice(crop_name: str) -> str:
    """Provide realistic agronomic recommendations for crop production."""
    advice_map = {
        "rice": "Requires heavy clayey or loamy soil capable of water retention. Needs flooded or saturated conditions. Split nitrogen application: 50% basal, 25% tillering, 25% panicle initiation.",
        "wheat": "Thrives in well-drained loamy to clay loam soils with neutral pH (6.0 - 7.5). Cool weather during vegetative stage and warm sunny weather during grain ripening.",
        "maize": "Prefers deep, rich, well-drained soils with pH 5.8-7.0. Highly responsive to nitrogen and zinc. Avoid waterlogged conditions.",
        "cotton": "Deep, fertile black cotton soils (Vertisols) or alluvial soils with pH 6.0-8.0. Needs warm days and sunny climate. Avoid excess moisture at boll opening.",
        "chickpea": "Well-drained light to medium soils with neutral to slightly alkaline pH (6.5-7.8). Fixes atmospheric nitrogen; requires high phosphorus (DAP) for root nodules.",
        "kidneybeans": "Light loamy soil with good organic matter, pH 5.5-6.5. Sensitive to water stagnation and saline conditions.",
        "banana": "Rich loamy soil with high organic matter, high moisture, and excellent drainage, pH 6.0-7.5. Heavy feeder of potassium (K) and nitrogen.",
        "mango": "Deep, well-drained loamy or alluvial soils with pH 5.5-7.5. Requires hot, dry period for flowering and fruit development.",
        "grapes": "Well-drained sandy loam to clay loam, pH 6.0-7.5. High potassium consumer for berry sweetness and yield.",
        "muskmelon": "Sandy loam soils with excellent drainage, pH 6.0-7.5. Warm dry climate with ample sunshine; low humidity during ripening prevents fungal rot.",
        "watermelon": "Sandy to sandy loam fertile soils, pH 6.0-7.0. Requires high sunshine, warm temperature, and moderate moisture.",
        "apple": "Loamy soil rich in organic matter with acidic to neutral pH (5.5-6.5). Requires chilling hours (below 7°C) during winter.",
        "orange": "Well-drained sandy loam or alluvial soil with pH 6.0-7.5. Sensitive to waterlogging and soil salinity.",
        "papaya": "Well-drained porous sandy loam soil, pH 6.0-7.0. Highly sensitive to waterlogging (causes collar rot).",
        "coconut": "Coastal sandy, alluvial, or red sandy loam soils, pH 5.2-8.0. Requires high humidity and warm tropical temperatures.",
        "coffee": "Deep, friable, porous volcanic or loamy soil rich in humus, pH 6.0-6.5. Grown under shade trees in subtropical/tropical highlands.",
        "jute": "New alluvial soils with standing river water availability, pH 6.4-7.4. Requires high humidity and warm monsoon climate.",
    }
    return advice_map.get(
        crop_name.lower(),
        "Ensure balanced NPK application based on regular soil health tests. Maintain organic matter with farmyard manure (FYM)."
    )


class MLService:
    def __init__(self):
        self.yield_model = None
        self.yield_preprocessor = None
        self.recommendation_model = None
        self.rec_preprocessor = None
        self.yield_metrics: dict = {}
        self.rec_metrics: dict = {}
        self.secondary_metrics: dict = {}

    def load_models(self):
        """Load all saved models. Called once at startup."""
        _load_soil_lookup()
        _load_crop_requirements()

        # Yield model
        y_model_path = SAVED_MODELS_DIR / "yield_model.pkl"
        y_prep_path  = SAVED_MODELS_DIR / "yield_preprocessor.pkl"
        if y_model_path.exists() and y_prep_path.exists():
            with open(y_model_path, "rb") as f:
                self.yield_model = pickle.load(f)
            with open(y_prep_path, "rb") as f:
                self.yield_preprocessor = pickle.load(f)
            logger.info("✅ Yield model loaded")
        else:
            logger.warning("⚠️  Yield model not found. Run train_yield_model.py first.")

        # Recommendation model
        r_model_path = SAVED_MODELS_DIR / "recommendation_model.pkl"
        r_prep_path  = SAVED_MODELS_DIR / "recommendation_preprocessor.pkl"
        if r_model_path.exists() and r_prep_path.exists():
            with open(r_model_path, "rb") as f:
                self.recommendation_model = pickle.load(f)
            with open(r_prep_path, "rb") as f:
                self.rec_preprocessor = pickle.load(f)
            logger.info("✅ Recommendation model loaded")
        else:
            logger.warning("⚠️  Recommendation model not found. Run train_recommendation_model.py first.")

        # Load metrics reports
        for attr, fname in [
            ("yield_metrics", "yield_model_metrics.json"),
            ("rec_metrics",   "recommendation_model_metrics.json"),
            ("secondary_metrics", "secondary_validation_metrics.json"),
        ]:
            p = REPORTS_DIR / fname
            if p.exists():
                with open(p) as f:
                    setattr(self, attr, json.load(f))

    def get_soil_for_state(self, state: str) -> dict:
        """Return soil values for a state from the lookup table."""
        return _SOIL_LOOKUP.get(state, {})

    def get_crop_requirements(self, crop_name: str) -> dict:
        """Return the ideal soil pH, NPK, and climate parameters for a requested crop."""
        key = crop_name.strip().lower()
        if key in _CROP_REQUIREMENTS:
            return {
                "success": True,
                "data": _CROP_REQUIREMENTS[key],
            }
        
        # Fuzzy / partial match
        for k, v in _CROP_REQUIREMENTS.items():
            if k in key or key in k:
                return {"success": True, "data": v}

        # Fallback for common unlisted crops
        common_profiles = {
            "wheat": {
                "crop": "Wheat",
                "soil_pH": {"optimal": 6.8, "range": "6.0 - 7.5", "min": 6.0, "max": 7.5},
                "nitrogen_N_kg_ha": {"optimal": 120.0, "range": "100 - 150 kg/ha", "min": 100, "max": 150},
                "phosphorus_P_kg_ha": {"optimal": 60.0, "range": "40 - 60 kg/ha", "min": 40, "max": 60},
                "potassium_K_kg_ha": {"optimal": 40.0, "range": "30 - 50 kg/ha", "min": 30, "max": 50},
                "climate": {
                    "temperature_c": "15 - 25 °C (cool vegetative, warm harvest)",
                    "humidity_pct": "50% - 70%",
                    "rainfall_mm": "75 - 100 mm total during growing season (requires irrigation)",
                },
                "agronomic_advice": "Rabi season crop. Requires fertile loamy soil. Apply 1/3 N + full P & K at sowing, remaining N in two splits at crown root initiation and tillering.",
            },
            "sugarcane": {
                "crop": "Sugarcane",
                "soil_pH": {"optimal": 6.5, "range": "6.0 - 7.5", "min": 5.5, "max": 8.0},
                "nitrogen_N_kg_ha": {"optimal": 250.0, "range": "200 - 300 kg/ha", "min": 200, "max": 300},
                "phosphorus_P_kg_ha": {"optimal": 80.0, "range": "60 - 100 kg/ha", "min": 60, "max": 100},
                "potassium_K_kg_ha": {"optimal": 120.0, "range": "80 - 150 kg/ha", "min": 80, "max": 150},
                "climate": {
                    "temperature_c": "20 - 35 °C (warm tropical)",
                    "humidity_pct": "70% - 85%",
                    "rainfall_mm": "1500 - 2500 mm (heavy water requirement)",
                },
                "agronomic_advice": "Deep, well-drained loam or clay loam. Long duration crop (10-12 months). Heavy consumer of organic matter and balanced NPK.",
            },
        }

        if key in common_profiles:
            return {"success": True, "data": common_profiles[key]}

        return {
            "success": False,
            "error": f"Crop '{crop_name}' not found. Available reference crops include: {', '.join(list(_CROP_REQUIREMENTS.keys()))}",
        }

    def resolve_location(self, location_str: str) -> tuple[str, float, float]:
        """Resolve a city/state name to (State_Name, Latitude, Longitude)."""
        loc_clean = location_str.strip().lower()
        if loc_clean in _LOCATION_COORDINATES:
            return _LOCATION_COORDINATES[loc_clean]

        for k, v in _LOCATION_COORDINATES.items():
            if k in loc_clean or loc_clean in k:
                return v

        # Fallback to Punjab as default representative state
        return ("Punjab", 30.90, 75.85)

    def predict_yield(self, input_data: dict) -> dict:
        """
        Run yield prediction.
        input_data must include all feature columns or None for auto-fill.
        Returns predicted yield in tonnes/ha and feature importance.
        """
        if self.yield_model is None or self.yield_preprocessor is None:
            raise RuntimeError("Yield model not loaded.")

        prep = self.yield_preprocessor

        # Auto-fill soil from state lookup
        state = input_data.get("state", "")
        soil = _SOIL_LOOKUP.get(state, {})
        for k in ["N", "P", "K", "pH"]:
            if input_data.get(k) is None:
                input_data[k] = soil.get(k, 0.0)

        # Validate required fields
        for col in prep.feature_cols:
            if col not in input_data or input_data[col] is None:
                input_data[col] = 0.0

        X = prep.transform_single(input_data)
        pred = float(self.yield_model.predict(X)[0])

        # Feature importance from saved metrics
        feat_imp = self.yield_metrics.get("feature_importance", {})

        return {
            "predicted_yield_tonnes_ha": round(pred, 4),
            "model_used": self.yield_metrics.get("best_model", type(self.yield_model).__name__),
            "feature_importance": feat_imp,
        }

    def recommend_crop(self, input_data: dict) -> dict:
        """
        Run crop recommendation.
        Returns recommended crop, top N crops with probabilities.
        """
        if self.recommendation_model is None or self.rec_preprocessor is None:
            raise RuntimeError("Recommendation model not loaded.")

        prep = self.rec_preprocessor
        X = prep.transform_single(input_data)

        classes = prep.get_classes()

        # Get probabilities if available
        if hasattr(self.recommendation_model, "predict_proba"):
            proba = self.recommendation_model.predict_proba(X)[0]
            top_indices = np.argsort(proba)[::-1][:5]
            top_crops = [
                {"crop": classes[i], "score": round(float(proba[i]), 4)}
                for i in top_indices
            ]
            recommended = classes[top_indices[0]]
        else:
            pred_idx = self.recommendation_model.predict(X)[0]
            recommended = classes[pred_idx]
            top_crops = [{"crop": recommended, "score": 1.0}]

        feat_imp = self.rec_metrics.get("feature_importance", {})

        return {
            "recommended_crop": recommended,
            "top_crops": top_crops,
            "model_used": self.rec_metrics.get("best_model", type(self.recommendation_model).__name__),
            "feature_importance": feat_imp,
        }

    def get_known_crops(self) -> list[str]:
        if self.yield_preprocessor:
            return self.yield_preprocessor.get_class_names("crop")
        return []

    def get_known_states(self) -> list[str]:
        if self.yield_preprocessor:
            return self.yield_preprocessor.get_class_names("state")
        return []

    def get_known_seasons(self) -> list[str]:
        if self.yield_preprocessor:
            return self.yield_preprocessor.get_class_names("season")
        return []

    def get_recommendable_crops(self) -> list[str]:
        if self.rec_preprocessor:
            return self.rec_preprocessor.get_classes()
        return []


# Singleton
ml_service = MLService()
ml_service.load_models()
