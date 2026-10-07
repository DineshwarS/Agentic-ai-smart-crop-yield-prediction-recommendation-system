"""
FastAPI Application Entry Point
"""

import logging
import os
from pathlib import Path
from contextlib import asynccontextmanager

from dotenv import load_dotenv

# Load environment variables from backend/.env or root .env
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import (
    agent_router,
    auth_router,
    performance_router,
    predictions_router,
    recommendation_router,
    weather_router,
    yield_router,
)
from .database.mongodb import connect_db, disconnect_db
from .services.ml_service import ml_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown events."""
    logger.info("Starting up: loading ML models ...")
    ml_service.load_models()
    logger.info("Connecting to MongoDB ...")
    await connect_db()
    yield
    logger.info("Shutting down ...")
    await disconnect_db()


app = FastAPI(
    title="Agentic AI Crop Yield Prediction & Recommendation API",
    description=(
        "A B.Tech final-year project API that combines historical agricultural data, "
        "live weather (Open-Meteo), Machine Learning, and Agentic AI (Gemini) to "
        "predict crop yield and recommend suitable crops."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow React dev server
origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(auth_router.router,           prefix="/api", tags=["Authentication"])
app.include_router(yield_router.router,          prefix="/api", tags=["Yield Prediction"])
app.include_router(recommendation_router.router, prefix="/api", tags=["Crop Recommendation"])
app.include_router(weather_router.router,        prefix="/api", tags=["Weather"])
app.include_router(agent_router.router,          prefix="/api", tags=["Agentic AI"])
app.include_router(predictions_router.router,    prefix="/api", tags=["Prediction History"])
app.include_router(performance_router.router,    prefix="/api", tags=["Model Performance"])


@app.get("/api/health", tags=["Health"])
async def health():
    yield_ready = ml_service.yield_model is not None
    rec_ready = ml_service.recommendation_model is not None
    return {
        "status": "ok",
        "yield_model_loaded": yield_ready,
        "recommendation_model_loaded": rec_ready,
        "message": "Agentic AI Crop System API is running",
    }
