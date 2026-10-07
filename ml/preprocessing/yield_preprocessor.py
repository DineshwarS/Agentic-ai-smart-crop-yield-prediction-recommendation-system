"""
Yield Data Preprocessor
Merges crop_yield.csv + state_soil_data.csv + state_weather_data.csv
Drops production (leakage), encodes categoricals, handles missing values.
"""

import os
import sys
import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
YIELD_DIR = PROJECT_ROOT / "Crop Yield Data with Soil and Weather Dataset"
SAVED_MODELS_DIR = PROJECT_ROOT / "ml" / "saved_models"
SAVED_MODELS_DIR.mkdir(exist_ok=True)


class YieldPreprocessor:
    """
    Merges and preprocesses the three yield-related datasets.
    Prevents target leakage by dropping 'production'.
    """

    CATEGORICAL_COLS = ["crop", "season", "state"]
    # production is excluded — leakage: yield = production / area
    # year is kept as numeric feature
    FEATURE_COLS = [
        "crop", "season", "state", "area", "fertilizer", "pesticide",
        "year", "N", "P", "K", "pH",
        "avg_temp_c", "total_rainfall_mm", "avg_humidity_percent",
    ]
    TARGET_COL = "yield"

    def __init__(self):
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="median")
        self.feature_cols = self.FEATURE_COLS.copy()
        self.is_fitted = False

    def load_raw_data(self) -> pd.DataFrame:
        """Load and merge all three source datasets."""
        logger.info("Loading crop_yield.csv ...")
        yield_df = pd.read_csv(YIELD_DIR / "crop_yield.csv")
        yield_df.columns = yield_df.columns.str.strip()
        yield_df["crop"] = yield_df["crop"].str.strip()
        yield_df["season"] = yield_df["season"].str.strip()
        yield_df["state"] = yield_df["state"].str.strip()

        logger.info("Loading state_soil_data.csv ...")
        soil_df = pd.read_csv(YIELD_DIR / "state_soil_data.csv")
        soil_df["state"] = soil_df["state"].str.strip()

        logger.info("Loading state_weather_data_1997_2020.csv ...")
        weather_df = pd.read_csv(YIELD_DIR / "state_weather_data_1997_2020.csv")
        weather_df["state"] = weather_df["state"].str.strip()

        # Merge soil (state-level static) — left join keeps all yield rows
        merged = yield_df.merge(soil_df, on="state", how="left")

        # Merge weather (state × year)
        merged = merged.merge(weather_df, on=["state", "year"], how="left")

        logger.info(f"Merged dataset: {merged.shape[0]} rows × {merged.shape[1]} columns")
        logger.info(f"Columns: {list(merged.columns)}")

        # Drop leakage column
        if "production" in merged.columns:
            merged.drop(columns=["production"], inplace=True)
            logger.info("Dropped 'production' column (leakage prevention)")

        return merged

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean merged dataframe."""
        # Drop rows with missing target
        before = len(df)
        df = df.dropna(subset=[self.TARGET_COL])
        logger.info(f"Dropped {before - len(df)} rows with missing target")

        # Drop duplicates
        before = len(df)
        df = df.drop_duplicates()
        logger.info(f"Dropped {before - len(df)} duplicate rows")

        # Remove non-positive yield (physically invalid)
        before = len(df)
        df = df[df[self.TARGET_COL] > 0]
        logger.info(f"Dropped {before - len(df)} rows with non-positive yield")

        # Log-transform extreme outliers in yield using IQR
        q1 = df[self.TARGET_COL].quantile(0.01)
        q99 = df[self.TARGET_COL].quantile(0.99)
        before = len(df)
        df = df[(df[self.TARGET_COL] >= q1) & (df[self.TARGET_COL] <= q99)]
        logger.info(f"Removed {before - len(df)} extreme yield outliers (1st–99th percentile)")

        return df

    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Fit encoders/scaler and transform. Returns (X, y)."""
        df = df.copy()

        # Encode categoricals
        for col in self.CATEGORICAL_COLS:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
            logger.info(f"Encoded '{col}': {len(le.classes_)} classes")

        X = df[self.feature_cols].values
        y = df[self.TARGET_COL].values

        # Impute missing (weather rows with no match)
        X = self.imputer.fit_transform(X)

        # Scale
        X = self.scaler.fit_transform(X)
        self.is_fitted = True
        return X, y

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted encoders/scaler."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit_transform first.")
        df = df.copy()
        for col in self.CATEGORICAL_COLS:
            le = self.label_encoders[col]
            df[col] = df[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
        X = df[self.feature_cols].values
        X = self.imputer.transform(X)
        X = self.scaler.transform(X)
        return X

    def transform_single(self, input_dict: dict) -> np.ndarray:
        """Transform a single prediction request dict."""
        row = {col: [input_dict.get(col, 0)] for col in self.feature_cols}
        df = pd.DataFrame(row)
        return self.transform(df)

    def save(self, path: Path | str):
        """Save preprocessor to pickle."""
        path = Path(path)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Preprocessor saved to {path}")

    @staticmethod
    def load(path: Path | str) -> "YieldPreprocessor":
        path = Path(path)
        with open(path, "rb") as f:
            return pickle.load(f)

    def get_class_names(self, col: str) -> list[str]:
        return list(self.label_encoders[col].classes_)
