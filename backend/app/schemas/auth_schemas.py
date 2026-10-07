"""Pydantic schemas for user authentication."""

from typing import Optional
from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=100, examples=["Shanthi"])
    email: EmailStr = Field(..., examples=["shanthi@example.com"])
    password: str = Field(..., min_length=6, max_length=128, examples=["SecurePass123"])


class LoginRequest(BaseModel):
    email: EmailStr = Field(..., examples=["shanthi@example.com"])
    password: str = Field(..., min_length=1, examples=["SecurePass123"])


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: "UserOut"


class UserOut(BaseModel):
    id: str
    name: str
    email: str


TokenResponse.model_rebuild()
