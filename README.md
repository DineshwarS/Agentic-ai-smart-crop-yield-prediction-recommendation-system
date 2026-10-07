# Agentic AI Based Smart Crop Yield Prediction and Recommendation System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-blue.svg)](https://reactjs.org/)
[![Vite](https://img.shields.io/badge/Vite-5.0-purple.svg)](https://vitejs.dev/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1-orange.svg)](https://xgboost.readthedocs.io/)
[![Zero IoT](https://img.shields.io/badge/IoT%20Sensors-None%20(Pure%20ML)-red.svg)](#important-architectural-clarification)

A production-grade, B.Tech Information Technology final-year project that predicts crop yields, recommends optimal crops, and delivers agricultural advice using historical Indian agricultural datasets, Open-Meteo live weather data, Machine Learning models, and Agentic AI tool orchestration.

---

## 🌟 Important Architectural Clarification

This system contains **NO IoT COMPONENT WHATSOEVER**. It does not use microcontrollers (ESP32/Arduino/Raspberry Pi), simulated sensor streams, or MQTT. Predictions and recommendations are grounded entirely in:
1. **Verified Historical Datasets** (1997–2020 agricultural records across 30 Indian states and 54 crops).
2. **Real-time Live Weather Integration** (via free Open-Meteo API).
3. **Scientifically Rigorous Machine Learning Pipelines** (XGBoost Regressor and Random Forest Classifier with zero target leakage).
4. **Agentic AI Orchestration Layer** (Gemini AI with native tool execution).

---

## 🚀 Key Features

* **🌾 Leakage-Free Yield Prediction Engine**: Multi-variable regression predicting tonnes/hectare. The `production` column is explicitly dropped to prevent target leakage ($Yield = Production / Area$).
* **🌱 Crop Recommendation Engine**: Multi-class classification mapping soil chemistry (N-P-K-pH) and climatic tolerances across 22 crops with top-3 alternative probability rankings.
* **☁️ Live Weather Aggregation**: Integrates Open-Meteo forecast and archive APIs with an aggregation layer that aligns daily metrics with model training definitions.
* **🤖 Agentic AI Agricultural Assistant**: Conversational agent that autonomously executes backend ML and weather tools to answer complex farm-management inquiries.
* **📊 Model Performance Dashboard**: Displays actual experimental evaluation metrics ($R^2$, RMSE, MAE, Accuracy, F1) computed from real dataset holdout splits and 5-fold cross-validation.
* **📜 MongoDB Prediction History**: Complete audit trail of predictions, weather snapshots, and model choices.

---

## 🔬 Experimental Evaluation Summary

All metrics are computed on actual datasets (no synthetic numbers):

### Task A — Crop Yield Prediction (Time-Aware Split)
* **Training Period**: 1997–2014 (14,025 samples)
* **Validation Period**: 2015–2017 (3,037 samples)
* **Holdout Test Period**: 2018–2020 (2,123 samples)

| Algorithm | Validation RMSE | Validation $R^2$ | Test RMSE | Test $R^2$ | Selected Status |
|---|---|---|---|---|---|
| **Linear Regression (Baseline)** | 10.1157 t/ha | 0.1034 | 10.7131 t/ha | 0.0901 | Baseline |
| **Random Forest Regressor** | 2.9558 t/ha | 0.9234 | 3.7062 t/ha | 0.8911 | Candidate |
| **XGBoost Regressor** | **2.7174 t/ha** | **0.9353** | **3.4924 t/ha** | **0.9033** | **BEST (Selected)** |

### Task B — Crop Recommendation (Multi-Class Classification)
* **Dataset**: 2,200 samples across 22 crop classes
* **Split**: 1,760 train / 440 test + 5-Fold Cross Validation

| Algorithm | Test Accuracy | Macro Precision | Macro Recall | Macro $F_1$ | 5-Fold CV $F_1$ | Selected Status |
|---|---|---|---|---|---|---|
| **Random Forest Classifier** | **99.55%** | **0.9955** | **0.9957** | **0.9955** | **0.9959 $\pm$ 0.0027** | **BEST (Selected)** |
| **XGBoost Classifier** | 99.32% | 0.9932 | 0.9937 | 0.9932 | 0.9955 $\pm$ 0.0025 | Candidate |

---

## 📁 Repository Structure

```
├── backend/
│   ├── app/
│   │   ├── api/             # FastAPI REST endpoints
│   │   ├── services/        # ML service, weather service, agent service
│   │   ├── database/        # Async MongoDB client (Motor)
│   │   ├── schemas/         # Pydantic validation models
│   │   └── main.py          # FastAPI application entry
│   ├── requirements.txt     # Backend Python dependencies
│   └── .env.example         # Environment template
│
├── frontend/
│   ├── src/
│   │   ├── components/      # Sidebar, Header, WeatherWidget, Charts
│   │   ├── pages/           # Dashboard, Yield, Rec, AI Agent, History, Perf
│   │   ├── api.js           # Axios API client
│   │   └── App.jsx          # React Router setup
│   ├── package.json         # Frontend npm dependencies
│   └── vite.config.js       # Vite build configuration
│
├── ml/
│   ├── data/                # Dataset inspection & characterization
│   ├── preprocessing/       # Leakage-free preprocessors & scalers
│   ├── training/            # Time-aware model training scripts
│   ├── evaluation/          # Secondary external validation scripts
│   └── saved_models/        # Serialized models (.pkl)
│
├── docs/                    # Architecture, ML pipeline, weather mapping docs
├── reports/                 # Auto-generated metric JSONs & visual plots
├── tests/                   # ML unit tests & FastAPI integration tests
└── README.md
```

---

## 🛠️ Quickstart Installation & Local Execution

### 1. Prerequisites
* Python 3.10+
* Node.js 18+ and npm
* MongoDB (Optional, local instance or MongoDB Atlas)

### 2. Backend Setup & Model Training
```bash
# 1. Install Python packages
pip install -r backend/requirements.txt

# 2. Train and serialize ML models (generates .pkl & real metric reports)
python ml/data/inspect_datasets.py
python ml/training/train_yield_model.py
python ml/training/train_recommendation_model.py
python ml/evaluation/validate_with_secondary.py

# 3. Configure environment variables (optional for Gemini Agent)
# Copy backend/.env.example to backend/.env and add your GEMINI_API_KEY

# 4. Start the FastAPI backend
cd backend
uvicorn app.main:app --reload --port 8000
```
Backend API will be accessible at `http://localhost:8000` with Swagger docs at `http://localhost:8000/docs`.

### 3. Frontend Setup
```bash
# In a separate terminal:
cd frontend
npm install
npm run dev
```
Frontend dashboard will launch at `http://localhost:5173`.

### 4. Running Automated Tests
```bash
# Run ML and API test suite
python -m unittest tests/ml/test_ml_models.py
python -m unittest tests/backend/test_api_endpoints.py
```

---

## 📄 License & Academic Integrity
Developed as a production-quality academic project. All experimental findings reflect actual runs against the provided agricultural datasets without data synthesis.
