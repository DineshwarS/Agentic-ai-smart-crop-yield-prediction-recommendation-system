"""Authentication API router (Register, Login, Me)."""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status

from ..database.auth_db import create_user, email_exists, find_user_by_email, find_user_by_id
from ..database.mongodb import get_db
from ..schemas.auth_schemas import LoginRequest, RegisterRequest, TokenResponse, UserOut
from ..services.auth_service import (
    create_access_token,
    get_current_user,
    hash_password,
    verify_password,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(req: RegisterRequest):
    """Register a new user in MongoDB and return access token."""
    db = get_db()
    if db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection is unavailable. Ensure MongoDB is running.",
        )

    if await email_exists(req.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An account with this email already exists.",
        )

    hashed_pw = hash_password(req.password)
    user_doc = {
        "name": req.name.strip(),
        "email": req.email.lower().strip(),
        "hashed_password": hashed_pw,
        "created_at": datetime.now(timezone.utc),
    }

    user_id = await create_user(user_doc)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register user account.",
        )

    token = create_access_token(user_id=user_id, email=req.email.lower().strip())
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        user=UserOut(id=user_id, name=req.name.strip(), email=req.email.lower().strip()),
    )


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest):
    """Authenticate user with email and password, return JWT token."""
    db = get_db()
    if db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection is unavailable. Ensure MongoDB is running.",
        )

    user = await find_user_by_email(req.email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    if not verify_password(req.password, user.get("hashed_password", "")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    token = create_access_token(user_id=user["_id"], email=user["email"])
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        user=UserOut(id=user["_id"], name=user["name"], email=user["email"]),
    )


@router.get("/me", response_model=UserOut)
async def me(current_user: dict = Depends(get_current_user)):
    """Return currently authenticated user information."""
    user = await find_user_by_id(current_user["sub"])
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )
    return UserOut(
        id=user["_id"],
        name=user.get("name", "User"),
        email=user.get("email", current_user.get("email", "")),
    )
