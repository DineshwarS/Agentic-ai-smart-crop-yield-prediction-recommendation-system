"""MongoDB async client using Motor."""

import logging
import os
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

logger = logging.getLogger(__name__)

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGODB_DB_NAME", "crop_yield_db")

_client: Optional[AsyncIOMotorClient] = None
_db: Optional[AsyncIOMotorDatabase] = None


async def connect_db():
    global _client, _db
    try:
        _client = AsyncIOMotorClient(MONGODB_URI, serverSelectionTimeoutMS=3000)
        _db = _client[DB_NAME]
        # Test connection
        await _client.admin.command("ping")
        logger.info(f"✅ MongoDB connected: {MONGODB_URI} → {DB_NAME}")
    except Exception as e:
        logger.warning(f"⚠️  MongoDB connection failed: {e}. Predictions will not be persisted.")
        _client = None
        _db = None


async def disconnect_db():
    global _client, _db
    if _client:
        _client.close()
        logger.info("MongoDB disconnected")
    _client = None
    _db = None


def get_db() -> Optional[AsyncIOMotorDatabase]:
    return _db


async def insert_prediction(record: dict) -> Optional[str]:
    """Insert a prediction record. Returns inserted ID or None if DB unavailable."""
    db = get_db()
    if db is None:
        return None
    try:
        result = await db.predictions.insert_one(record)
        return str(result.inserted_id)
    except Exception as e:
        logger.error(f"MongoDB insert error: {e}")
        return None


async def get_predictions(limit: int = 50) -> list[dict]:
    """Fetch recent prediction records."""
    db = get_db()
    if db is None:
        return []
    try:
        cursor = db.predictions.find().sort("timestamp", -1).limit(limit)
        records = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            records.append(doc)
        return records
    except Exception as e:
        logger.error(f"MongoDB query error: {e}")
        return []


async def get_prediction_by_id(prediction_id: str) -> Optional[dict]:
    """Fetch a single prediction by ID."""
    db = get_db()
    if db is None:
        return None
    try:
        from bson import ObjectId
        doc = await db.predictions.find_one({"_id": ObjectId(prediction_id)})
        if doc:
            doc["_id"] = str(doc["_id"])
        return doc
    except Exception as e:
        logger.error(f"MongoDB fetch error: {e}")
        return None
