"""
Crop Recommendation Preprocessor
Handles Crop_recommendation.csv feature encoding and scaling.
"""

import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
SAVED_MODELS_DIR = PROJECT_ROOT / "ml" / "saved_models"
SAVED_MODELS_DIR.mkdir(exist_ok=True)


class RecommendationPreprocessor:
    """
    Preprocessor for the crop recommendation dataset.
    Features: N, P, K, temperature, humidity, ph, rainfall
    Target: label (crop class)
    """

    FEATURE_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    TARGET_COL = "label"

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False

    def load_data(self, csv_path: Path | str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        logger.info(f"Loaded recommendation dataset: {df.shape}")
        return df

    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        df = df.copy()

        # Drop duplicates and nulls
        df = df.dropna()
        df = df.drop_duplicates()
        logger.info(f"After cleaning: {df.shape}")

        X = df[self.FEATURE_COLS].values
        y_raw = df[self.TARGET_COL].values

        # Encode target labels
        y = self.label_encoder.fit_transform(y_raw)
        logger.info(f"Encoded {len(self.label_encoder.classes_)} crop classes")

        # Scale features
        X = self.scaler.fit_transform(X)
        self.is_fitted = True
        return X, y

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Preprocessor not fitted.")
        X = df[self.FEATURE_COLS].values
        return self.scaler.transform(X)

    def transform_single(self, input_dict: dict) -> np.ndarray:
        row = {col: [input_dict.get(col, 0)] for col in self.FEATURE_COLS}
        df = pd.DataFrame(row)
        return self.transform(df)

    def decode_label(self, encoded: int) -> str:
        return self.label_encoder.classes_[encoded]

    def decode_labels(self, encoded_array: np.ndarray) -> list[str]:
        return [self.label_encoder.classes_[e] for e in encoded_array]

    def get_classes(self) -> list[str]:
        return list(self.label_encoder.classes_)

    def save(self, path: Path | str):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Recommendation preprocessor saved to {path}")

    @staticmethod
    def load(path: Path | str) -> "RecommendationPreprocessor":
        with open(path, "rb") as f:
            return pickle.load(f)
