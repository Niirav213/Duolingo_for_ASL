"""Lesson and game schemas."""
from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class LessonBase(BaseModel):
    """Base lesson schema."""
    title: str
    description: str
    sign_name: str
    difficulty_level: int


class LessonCreate(LessonBase):
    """Lesson creation schema."""
    pass


class LessonResponse(LessonBase):
    """Lesson response schema."""
    id: int

    class Config:
        from_attributes = True


class UserProgressResponse(BaseModel):
    """User progress response schema."""
    lesson_id: int
    xp_gained: int
    completed_at: datetime
    current_level: int
    total_xp: int

    class Config:
        from_attributes = True


class GameSessionStart(BaseModel):
    """Start game session schema."""
    lesson_id: int


class GameSessionResponse(BaseModel):
    """Game session response schema."""
    id: int
    lesson_id: int
    score: int
    accuracy: float
    duration_seconds: int
    completed: bool
    created_at: datetime

    class Config:
        from_attributes = True


class GestureDetectionRequest(BaseModel):
    """Gesture detection request schema."""
    image_data: str  # base64 encoded image
    lesson_id: int


class GestureDetectionResponse(BaseModel):
    """Gesture detection response schema."""
    detected_sign: str
    confidence: float
    correct: bool
    xp_gained: int
