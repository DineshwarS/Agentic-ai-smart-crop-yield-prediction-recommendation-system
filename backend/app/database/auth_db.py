"""MongoDB CRUD operations for user authentication."""

import logging
from typing import Optional

from .mongodb import get_db

logger = logging.getLogger(__name__)

USERS_COLLECTION = "users"


async def create_user(user_doc: dict) -> Optional[str]:
    """Insert a new user. Returns the inserted ObjectId string or None on error."""
    db = get_db()
    if db is None:
        logger.error("MongoDB not available — cannot create user.")
        return None
    try:
        result = await db[USERS_COLLECTION].insert_one(user_doc)
        return str(result.inserted_id)
    except Exception as e:
        logger.error(f"create_user error: {e}")
        return None


async def find_user_by_email(email: str) -> Optional[dict]:
    """Find a user by email. Returns the raw MongoDB document or None."""
    db = get_db()
    if db is None:
        return None
    try:
        doc = await db[USERS_COLLECTION].find_one({"email": email.lower()})
        if doc:
            doc["_id"] = str(doc["_id"])
        return doc
    except Exception as e:
        logger.error(f"find_user_by_email error: {e}")
        return None


async def find_user_by_id(user_id: str) -> Optional[dict]:
    """Find a user by string ObjectId. Returns document without hashed_password."""
    db = get_db()
    if db is None:
        return None
    try:
        from bson import ObjectId
        doc = await db[USERS_COLLECTION].find_one({"_id": ObjectId(user_id)})
        if doc:
            doc["_id"] = str(doc["_id"])
        return doc
    except Exception as e:
        logger.error(f"find_user_by_id error: {e}")
        return None


async def email_exists(email: str) -> bool:
    """Return True if the email is already registered."""
    db = get_db()
    if db is None:
        return False
    try:
        count = await db[USERS_COLLECTION].count_documents({"email": email.lower()})
        return count > 0
    except Exception as e:
        logger.error(f"email_exists error: {e}")
        return False
