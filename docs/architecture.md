# System Architecture

## 1. High-Level Architecture Overview

The **Agentic AI Based Smart Crop Yield Prediction and Recommendation System** integrates historical agricultural datasets, farmer-provided inputs, live Open-Meteo weather intelligence, Machine Learning regression/classification pipelines, and an Agentic AI reasoning layer.

```
+-------------------------------------------------------------------------------+
|                               Farmer (Client)                                 |
+-------------------------------------------------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                       Frontend: React + Vite Dashboard                        |
|   - Dashboard (Live Weather, Quick Forecasts, Analytics)                      |
|   - Yield Prediction (Leakage-Free Form, Model Explainability, SHAP)          |
|   - Crop Recommendation (Soil Profile, Probability Ranking, Agronomy)        |
|   - AI Agricultural Agent (Conversational Tool-Calling Interface)             |
|   - Prediction History (MongoDB Audit Trail)                                  |
|   - Model Performance (Actual Experimental Benchmarks & Validations)          |
+-------------------------------------------------------------------------------+
                                        |  REST API (JSON)
                                        v
+-------------------------------------------------------------------------------+
|                             FastAPI Backend Core                              |
|   +-----------------------+----------------------+------------------------+   |
|   |   Yield Router        | Recommendation Router|  Agent Chat Router     |   |
|   +-----------------------+----------------------+------------------------+   |
|   |   Weather Router      | Performance Router   |  History Router        |   |
|   +-----------------------+----------------------+------------------------+   |
+-------------------------------------------------------------------------------+
         |                         |                         |              |
         v                         v                         v              v
+------------------+     +--------------------+   +-------------------+  +------+
| Weather Service  |     | Machine Learning   |   | Agentic AI        |  | DB   |
| (Open-Meteo API) |     | Service (Inference)|   | (Gemini Agent)    |  |Motor |
| - Live 7-Day     |     | - XGBoost Yield    |   | - Tool Calling:   |  |Mongo |
| - Annual Archive |     | - Random Forest Rec|   |   get_weather     |  |      |
| - Aggregation    |     | - Preprocessors    |   |   predict_yield   |  |Audit |
+------------------+     +--------------------+   |   recommend_crop  |  |Log   |
                                                  |   compare_crops   |  +------+
                                                  +-------------------+
```

---

## 2. Component Design

### 2.1 Backend Layer (FastAPI)
- **Asynchronous Lifespan Management**: ML models (`yield_model.pkl`, `recommendation_model.pkl`) and preprocessor pipelines are loaded once at startup into memory for sub-millisecond inference.
- **Pydantic Validation**: Strict type-checking, bounds validation, and custom validators for inputs like soil pH [0, 14], N-P-K nutrient values, and geocoordinates.
- **CORS Middleware**: Secure browser cross-origin requests from React dev and production servers.

### 2.2 Machine Learning Inference Layer
- **Crop Yield Predictor**:
  - Algorithm: **XGBoost Regressor** (auto-selected over Random Forest & Linear Regression based on lowest validation RMSE: 2.7174 t/ha, $R^2 = 0.9353$).
  - Target: Crop Yield (tonnes/ha).
  - Features: `crop`, `season`, `state`, `area`, `fertilizer`, `pesticide`, `year`, `N`, `P`, `K`, `pH`, `avg_temp_c`, `total_rainfall_mm`, `avg_humidity_percent`.
  - Leakage Guard: `production` is strictly excluded.
- **Crop Recommender**:
  - Algorithm: **Random Forest Classifier** (auto-selected over XGBoost based on test accuracy: 99.55%, Macro F1: 0.9955).
  - Target: Crop Class (22 multi-class categories).
  - Features: `N`, `P`, `K`, `temperature`, `humidity`, `ph`, `rainfall`.

### 2.3 Weather Integration Service
- Connected to **Open-Meteo Forecast & Historical Archive API** (Free, open-access, zero API key requirement).
- Solves domain-mismatch via a dedicated aggregation layer: converts live/daily precipitation and temperature into seasonal/annual estimates matching model feature semantics.

### 2.4 Agentic AI Orchestration (Gemini)
- Uses Gemini 1.5 with native tool-calling (`function_declarations`).
- Autonomous tool execution loop:
  1. User asks question in natural language.
  2. Agent analyzes intent and required parameters.
  3. Agent executes backend tools (`get_weather`, `predict_yield`, `recommend_crop`, `compare_crops`).
  4. Agent grounds response strictly on tool outputs (No hallucinations, no fabricated numbers).
  5. Generates farmer-friendly agricultural explanations.

### 2.5 Database Persistence Layer (MongoDB)
- Persists all prediction requests, outputs, weather snapshots, and timestamps.
- Async I/O powered by `motor`.
- Graceful degradation: If MongoDB is offline, system continues full ML prediction and chat functionality without crashing.

### 2.6 Frontend Layer (React + Vite)
- Modern SPA design with dark-mode aesthetic, custom glassmorphism, fluid micro-animations, and responsive layouts.
- Component architecture with separation of concerns between API client, visual charts, and state containers.
- Zero fake metrics: All stats on the Model Performance page are fed directly from generated JSON evaluation reports.
