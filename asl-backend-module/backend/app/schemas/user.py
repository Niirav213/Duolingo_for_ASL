"""User schemas for request/response validation."""
from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional


class UserBase(BaseModel):
    """Base user schema."""
    username: str
    email: EmailStr


class UserCreate(UserBase):
    """User creation schema."""
    password: str


class UserLogin(BaseModel):
    """User login schema."""
    username: str
    password: str


class UserResponse(UserBase):
    """User response schema."""
    id: int
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    """Token response schema."""
    access_token: str
    refresh_token: str
    token_type: str


class TokenRefresh(BaseModel):
    """Token refresh schema."""
    refresh_token: str
