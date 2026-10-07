# Dataset Analysis & Preprocessing Report

## 1. Overview of Inspected Datasets

All five dataset files provided in the workspace were inspected and statistically characterized prior to model training.

---

## 2. Dataset Specifics

### 2.1 Primary Crop Yield Dataset (`crop_yield.csv`)
* **Shape**: 19,689 rows × 9 columns
* **Target Feature**: `yield` (expressed in tonnes per hectare)
* **Raw Columns**: `crop`, `year`, `season`, `state`, `area`, `production`, `fertilizer`, `pesticide`, `yield`
* **Temporal Span**: 1997 to 2020 (24 years of records across 30 Indian states and 54 distinct crop types)
* **Target Leakage Remediation**: The column `production` was strictly excluded from all training feature vectors because $Yield = Production / Area$, which constitutes an algebraic identity and would lead to trivial artificial leakage.
* **Cleaning Protocol**:
  - Removed 112 records with non-positive yield values.
  - Trimmed 392 extreme statistical outliers outside the 1st–99th percentile range to prevent gradient distortion during tree boosting.
  - Resulting clean training base: 19,185 observations.

### 2.2 State-Level Soil Chemistry Dataset (`state_soil_data.csv`)
* **Shape**: 30 rows × 5 columns
* **Columns**: `state`, `N` (Nitrogen), `P` (Phosphorus), `K` (Potassium), `pH`
* **Role**: Merged with yield records by `state` to provide static regional soil baseline chemistry.

### 2.3 State-Level Weather Historical Dataset (`state_weather_data_1997_2020.csv`)
* **Shape**: 720 rows × 5 columns
* **Columns**: `state`, `year`, `avg_temp_c`, `total_rainfall_mm`, `avg_humidity_percent`
* **Role**: Merged with yield records on composite key `[state, year]` to supply annual macroeconomic climate aggregates.

### 2.4 Crop Recommendation Dataset (`Crop_recommendation.csv`)
* **Shape**: 2,200 rows × 8 columns
* **Target**: `label` (22 unique multi-class crops: rice, maize, chickpea, kidneybeans, pigeonpeas, mothbeans, mungbean, blackgram, lentil, pomegranate, banana, mango, grapes, watermelon, muskmelon, apple, orange, papaya, coconut, cotton, jute, coffee)
* **Features**: `N`, `P`, `K`, `temperature` (°C), `humidity` (%), `ph`, `rainfall` (mm).
* **Balanced Profile**: Exactly 100 balanced observations per crop class.

### 2.5 Secondary Benchmark Validation Dataset (`Crops yield and Historical Dataset.csv`)
* **Shape**: 50,765 rows × 20 columns
* **Target Column**: `Yield_kg_per_ha`
* **Compatibility Analysis**:
  - **Unit Alignment**: Raw yield in kg/ha was transformed via $Yield_{tonnes/ha} = Yield_{kg/ha} / 1000$.
  - Evaluated 25,951 compatible records for cross-dataset generalization benchmarking.

---

## 3. Preprocessing & Encoding Pipeline
* Categorical columns (`crop`, `season`, `state`) are encoded via `sklearn.preprocessing.LabelEncoder`.
* Numerical features are normalized via `StandardScaler` fitted strictly on training data subsets to prevent data contamination across splits.
* Missing entries are imputed using median replacement via `SimpleImputer`.
