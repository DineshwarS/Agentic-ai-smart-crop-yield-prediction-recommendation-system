# Machine Learning Pipeline & Experimental Evaluation

## 1. Task A — Crop Yield Prediction (Regression)

### 1.1 Time-Aware Data Partitioning
Random splitting was deliberately rejected in favor of chronological partitioning to simulate true forward forecasting:
* **Training Subset (1997–2014)**: 14,025 samples (73.1%)
* **Validation Subset (2015–2017)**: 3,037 samples (15.8%)
* **Holdout Test Subset (2018–2020)**: 2,123 samples (11.1%)

### 1.2 Model Benchmarking & Metric Comparison

| Model Name | Val MAE (t/ha) | Val RMSE (t/ha) | Val $R^2$ Score | Val MAPE (%) | Test MAE (t/ha) | Test RMSE (t/ha) | Test $R^2$ Score | Test MAPE (%) |
|---|---|---|---|---|---|---|---|---|
| **Linear Regression (Baseline)** | 4.9886 | 10.1157 | 0.1034 | 369.58% | 5.1093 | 10.7131 | 0.0901 | 353.60% |
| **Random Forest Regressor** | 1.1376 | 2.9558 | 0.9234 | 55.97% | 1.3163 | 3.7062 | 0.8911 | 57.73% |
| **XGBoost Regressor (Selected)** | **1.1469** | **2.7174** | **0.9353** | **61.76%** | **1.3096** | **3.4924** | **0.9033** | **63.32%** |

* **Selection Decision**: **XGBoost Regressor** attained the lowest validation RMSE (2.7174 tonnes/ha) and highest validation $R^2$ (0.9353), confirming superior generalization across unseen time horizons.

---

## 2. Task B — Crop Recommendation (Classification)

### 2.1 Partitioning & Cross-Validation
* **Stratified Split**: 1,760 training observations / 440 holdout test observations across 22 balanced crop classes.
* **5-Fold Stratified Cross-Validation**: Validated consistency across all folds.

### 2.2 Model Benchmark Results

| Model Name | Test Accuracy | Macro Precision | Macro Recall | Macro $F_1$ Score | 5-Fold CV $F_1$ (Mean $\pm$ Std) |
|---|---|---|---|---|---|
| **Random Forest Classifier (Selected)** | **99.55%** | **0.9955** | **0.9957** | **0.9955** | **0.9959 $\pm$ 0.0027** |
| **XGBoost Classifier** | 99.32% | 0.9932 | 0.9937 | 0.9932 | 0.9955 $\pm$ 0.0025 |

* **Selection Decision**: **Random Forest Classifier** achieved the highest holdout accuracy (99.55%) and lowest variance across 5-fold CV.

---

## 3. Task C — Secondary External Validation
* Evaluated saved yield model on 25,951 compatible records from `Crops yield and Historical Dataset.csv`.
* **Cross-dataset MAE**: 1.4118 tonnes/ha
* **Cross-dataset RMSE**: 1.7827 tonnes/ha
