"""
Dataset Inspection Script
Analyzes all provided datasets and generates a comprehensive report.
Run: python ml/data/inspect_datasets.py
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Paths (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATASETS_DIR = PROJECT_ROOT
YIELD_DIR = PROJECT_ROOT / "Crop Yield Data with Soil and Weather Dataset"
VALIDATE_DIR = PROJECT_ROOT / "Validate test model dataset"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def analyze_dataframe(df: pd.DataFrame, name: str) -> dict:
    """Run full analysis on a dataframe and return results dict."""
    report = {
        "name": name,
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": df.isnull().sum().to_dict(),
        "missing_percent": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "numerical_columns": list(df.select_dtypes(include=[np.number]).columns),
        "categorical_columns": list(df.select_dtypes(include=["object", "category"]).columns),
    }

    # Numerical stats
    numerical_stats = {}
    for col in report["numerical_columns"]:
        s = df[col].describe()
        numerical_stats[col] = {
            "min": float(s["min"]) if not np.isnan(s["min"]) else None,
            "max": float(s["max"]) if not np.isnan(s["max"]) else None,
            "mean": float(s["mean"]) if not np.isnan(s["mean"]) else None,
            "std": float(s["std"]) if not np.isnan(s["std"]) else None,
            "median": float(df[col].median()),
        }
        # IQR-based outlier count
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        outlier_count = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
        numerical_stats[col]["outlier_count_iqr"] = outlier_count
    report["numerical_stats"] = numerical_stats

    # Categorical unique values
    categorical_stats = {}
    for col in report["categorical_columns"]:
        categorical_stats[col] = {
            "unique_count": int(df[col].nunique()),
            "top_values": df[col].value_counts().head(10).to_dict(),
        }
    report["categorical_stats"] = categorical_stats

    return report


def print_report(report: dict):
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  Dataset: {report['name']}")
    print(sep)
    print(f"  Shape        : {report['shape']['rows']} rows × {report['shape']['columns']} columns")
    print(f"  Columns      : {', '.join(report['columns'])}")
    print(f"  Duplicates   : {report['duplicate_rows']}")
    print()
    print("  Missing Values:")
    for col, count in report["missing_values"].items():
        pct = report["missing_percent"][col]
        if count > 0:
            print(f"    {col}: {count} ({pct}%)")
    print()
    print(f"  Numerical Columns  : {', '.join(report['numerical_columns'])}")
    print(f"  Categorical Columns: {', '.join(report['categorical_columns'])}")
    print()
    print("  Numerical Summary:")
    for col, stats in report["numerical_stats"].items():
        print(f"    {col}: min={stats['min']:.3f}, max={stats['max']:.3f}, "
              f"mean={stats['mean']:.3f}, outliers={stats['outlier_count_iqr']}")
    print()
    if report["categorical_stats"]:
        print("  Categorical Summary:")
        for col, stats in report["categorical_stats"].items():
            top = list(stats["top_values"].keys())[:5]
            print(f"    {col}: {stats['unique_count']} unique | top: {top}")


def main():
    all_reports = {}

    # -------------------------------------------------------------------------
    # 1. Crop Yield (primary)
    # -------------------------------------------------------------------------
    print("\n[1] Loading crop_yield.csv ...")
    yield_df = pd.read_csv(YIELD_DIR / "crop_yield.csv")
    r1 = analyze_dataframe(yield_df, "crop_yield.csv")
    r1["target_column"] = "yield"
    r1["task"] = "Regression — Crop Yield Prediction"
    r1["leakage_warning"] = ("'production' column = production/area = yield → "
                             "MUST be excluded from features")
    r1["year_range"] = f"{int(yield_df['year'].min())}–{int(yield_df['year'].max())}"
    r1["unique_crops"] = int(yield_df["crop"].nunique())
    r1["unique_states"] = int(yield_df["state"].nunique())
    print_report(r1)
    print(f"  [!] LEAKAGE WARNING: {r1['leakage_warning']}")
    print(f"  Year range: {r1['year_range']}")
    print(f"  Unique crops: {r1['unique_crops']}, Unique states: {r1['unique_states']}")
    all_reports["crop_yield"] = r1

    # -------------------------------------------------------------------------
    # 2. State Soil Data
    # -------------------------------------------------------------------------
    print("\n[2] Loading state_soil_data.csv ...")
    soil_df = pd.read_csv(YIELD_DIR / "state_soil_data.csv")
    r2 = analyze_dataframe(soil_df, "state_soil_data.csv")
    r2["note"] = "Static per-state soil averages — join on 'state' column"
    print_report(r2)
    all_reports["state_soil"] = r2

    # -------------------------------------------------------------------------
    # 3. State Weather Data
    # -------------------------------------------------------------------------
    print("\n[3] Loading state_weather_data_1997_2020.csv ...")
    weather_df = pd.read_csv(YIELD_DIR / "state_weather_data_1997_2020.csv")
    r3 = analyze_dataframe(weather_df, "state_weather_data_1997_2020.csv")
    r3["note"] = ("Annual aggregates: avg_temp_c=annual average °C, "
                  "total_rainfall_mm=annual total mm, avg_humidity_percent=annual avg %")
    r3["join_key"] = ["state", "year"]
    print_report(r3)
    all_reports["state_weather"] = r3

    # -------------------------------------------------------------------------
    # 4. Crop Recommendation
    # -------------------------------------------------------------------------
    print("\n[4] Loading Crop_recommendation.csv ...")
    rec_df = pd.read_csv(DATASETS_DIR / "Crop_recommendation.csv")
    r4 = analyze_dataframe(rec_df, "Crop_recommendation.csv")
    r4["target_column"] = "label"
    r4["task"] = "Classification — Crop Recommendation"
    r4["unique_crops"] = int(rec_df["label"].nunique())
    r4["class_distribution"] = rec_df["label"].value_counts().to_dict()
    r4["note"] = ("Features: N, P, K, temperature, humidity, ph, rainfall. "
                  "rainfall here is likely seasonal/growing-season mm — NOT annual total.")
    print_report(r4)
    print(f"  Unique crop labels: {r4['unique_crops']}")
    print(f"  Classes: {list(r4['class_distribution'].keys())[:10]} ...")
    all_reports["crop_recommendation"] = r4

    # -------------------------------------------------------------------------
    # 5. Validation Dataset
    # -------------------------------------------------------------------------
    print("\n[5] Loading 'Crops yield and Historical Dataset.csv' ...")
    val_df = pd.read_csv(VALIDATE_DIR / "Crops yield and Historical Dataset.csv")
    r5 = analyze_dataframe(val_df, "Crops yield and Historical Dataset.csv")
    r5["target_column"] = "Yield_kg_per_ha"
    r5["task"] = "External Validation Only (NOT for training)"
    r5["unit_warning"] = ("Yield unit = kg/ha. Primary dataset uses tonnes/ha. "
                          "Apply / 1000 conversion before cross-dataset comparison.")
    r5["year_range"] = f"{int(val_df['Year'].min())}–{int(val_df['Year'].max())}"
    r5["unique_crops"] = int(val_df["Crop"].nunique())
    r5["unique_states"] = int(val_df["State Name"].nunique())
    print_report(r5)
    print(f"  [!] UNIT WARNING: {r5['unit_warning']}")
    print(f"  Year range: {r5['year_range']}")
    all_reports["validation"] = r5

    # -------------------------------------------------------------------------
    # Compatibility check between datasets
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  COMPATIBILITY ANALYSIS")
    print("=" * 70)

    # Check yield column compatibility
    print("\n  Primary yield (tonnes/ha) vs Validation yield (kg/ha):")
    print("  -> Unit conversion required: validation_yield / 1000 = tonnes/ha")

    # Check shared crops
    primary_crops_raw = set(yield_df["crop"].str.strip().str.lower())
    val_crops_raw = set(val_df["Crop"].str.strip().str.lower())
    common_crops = primary_crops_raw & val_crops_raw
    print(f"\n  Crops in both primary & validation: {len(common_crops)}")
    if common_crops:
        print(f"  Common crops: {sorted(list(common_crops))[:10]}")

    # Weather feature mapping
    print("\n  Weather Feature Mapping (Training -> Live API):")
    print("  avg_temp_c (annual avg C)       -> Open-Meteo: temperature_2m mean (aggregate)")
    print("  total_rainfall_mm (annual total)  -> Open-Meteo: precipitation_sum (aggregate)")
    print("  avg_humidity_percent (annual avg) -> Open-Meteo: relative_humidity_2m (aggregate)")

    # -------------------------------------------------------------------------
    # Save report JSON
    # -------------------------------------------------------------------------
    report_path = REPORTS_DIR / "dataset_analysis_report.json"
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        else:
            return obj

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(make_serializable(all_reports), f, indent=2)
    print(f"\n  [+] Full analysis saved to: {report_path}")

    # Save text summary
    text_path = REPORTS_DIR / "dataset_analysis_summary.txt"
    with open(text_path, "w", encoding="utf-8") as f:
        f.write("DATASET ANALYSIS SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        for name, r in all_reports.items():
            f.write(f"Dataset: {r['name']}\n")
            f.write(f"  Rows: {r['shape']['rows']}, Columns: {r['shape']['columns']}\n")
            f.write(f"  Columns: {', '.join(r['columns'])}\n")
            f.write(f"  Duplicates: {r['duplicate_rows']}\n")
            missing = {k: v for k, v in r['missing_values'].items() if v > 0}
            f.write(f"  Missing: {missing or 'None'}\n")
            if "leakage_warning" in r:
                f.write(f"  [!] LEAKAGE: {r['leakage_warning']}\n")
            if "unit_warning" in r:
                f.write(f"  [!] UNIT: {r['unit_warning']}\n")
            f.write("\n")
    print(f"  [+] Text summary saved to: {text_path}\n")


if __name__ == "__main__":
    main()
