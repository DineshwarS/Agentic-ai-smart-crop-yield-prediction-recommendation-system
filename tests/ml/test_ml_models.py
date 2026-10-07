"""
Unit and integration tests for ML models and preprocessors.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


from ml.preprocessing.yield_preprocessor import YieldPreprocessor
from ml.preprocessing.recommendation_preprocessor import RecommendationPreprocessor
from backend.app.services.ml_service import ml_service


class TestYieldPreprocessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prep_path = PROJECT_ROOT / "ml" / "saved_models" / "yield_preprocessor.pkl"
        if cls.prep_path.exists():
            cls.preprocessor = YieldPreprocessor.load(cls.prep_path)
        else:
            cls.preprocessor = None

    def test_preprocessor_loaded(self):
        self.assertIsNotNone(self.preprocessor, "Yield preprocessor should exist")

    def test_no_production_in_features(self):
        """Verify leakage prevention: production must NOT be in feature columns."""
        self.assertNotIn(
            "production",
            YieldPreprocessor.FEATURE_COLS,
            "Target leakage: 'production' must not be in feature columns",
        )

    def test_transform_single(self):
        if not self.preprocessor:
            self.skipTest("Preprocessor not saved")

        sample = {
            "crop": "Rice",
            "season": "Kharif",
            "state": "Punjab",
            "area": 10.0,
            "fertilizer": 1500.0,
            "pesticide": 25.0,
            "year": 2024,
            "N": 150.0,
            "P": 50.0,
            "K": 40.0,
            "pH": 8.0,
            "avg_temp_c": 25.0,
            "total_rainfall_mm": 800.0,
            "avg_humidity_percent": 65.0,
        }
        X = self.preprocessor.transform_single(sample)
        self.assertEqual(X.shape, (1, len(self.preprocessor.feature_cols)))
        self.assertFalse(np.isnan(X).any(), "Transformed features should not contain NaN")


class TestRecommendationPreprocessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prep_path = PROJECT_ROOT / "ml" / "saved_models" / "recommendation_preprocessor.pkl"
        if cls.prep_path.exists():
            cls.preprocessor = RecommendationPreprocessor.load(cls.prep_path)
        else:
            cls.preprocessor = None

    def test_preprocessor_loaded(self):
        self.assertIsNotNone(self.preprocessor, "Recommendation preprocessor should exist")

    def test_transform_single(self):
        if not self.preprocessor:
            self.skipTest("Preprocessor not saved")

        sample = {
            "N": 90,
            "P": 42,
            "K": 43,
            "temperature": 20.87,
            "humidity": 82.0,
            "ph": 6.5,
            "rainfall": 202.93,
        }
        X = self.preprocessor.transform_single(sample)
        self.assertEqual(X.shape, (1, len(self.preprocessor.FEATURE_COLS)))
        self.assertFalse(np.isnan(X).any(), "Transformed features should not contain NaN")


class TestMLService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ml_service.load_models()

    def test_yield_prediction(self):
        sample = {
            "crop": "Rice",
            "season": "Kharif",
            "state": "Punjab",
            "area": 10.0,
            "fertilizer": 1500.0,
            "pesticide": 25.0,
            "year": 2024,
            "N": 150.0,
            "P": 50.0,
            "K": 40.0,
            "pH": 8.0,
            "avg_temp_c": 25.0,
            "total_rainfall_mm": 800.0,
            "avg_humidity_percent": 65.0,
        }
        res = ml_service.predict_yield(sample)
        self.assertIn("predicted_yield_tonnes_ha", res)
        self.assertGreater(res["predicted_yield_tonnes_ha"], 0)
        self.assertIn("model_used", res)

    def test_crop_recommendation(self):
        sample = {
            "N": 90,
            "P": 42,
            "K": 43,
            "temperature": 20.87,
            "humidity": 82.0,
            "ph": 6.5,
            "rainfall": 202.93,
        }
        res = ml_service.recommend_crop(sample)
        self.assertIn("recommended_crop", res)
        self.assertIsInstance(res["recommended_crop"], str)
        self.assertIn("top_crops", res)
        self.assertGreater(len(res["top_crops"]), 0)


if __name__ == "__main__":
    unittest.main()
