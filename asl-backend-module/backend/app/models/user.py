"""User and Streak SQLAlchemy models."""
from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from app.db.base import Base


class User(Base):
    __tablename__ = "users"

    id               = Column(Integer, primary_key=True, index=True)
    username         = Column(String(50), unique=True, nullable=False, index=True)
    email            = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password  = Column(String(255), nullable=False)
    is_active        = Column(Boolean, default=True)
    created_at       = Column(DateTime, default=datetime.utcnow)
    updated_at       = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    progress         = relationship("UserProgress", back_populates="user")
    game_sessions    = relationship("GameSession", back_populates="user")
    streak           = relationship("Streak", back_populates="user", uselist=False)


class Streak(Base):
    __tablename__ = "streaks"

    id                  = Column(Integer, primary_key=True, index=True)
    user_id             = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False, index=True)
    current_streak      = Column(Integer, default=0)
    longest_streak      = Column(Integer, default=0)
    last_activity_date  = Column(DateTime, nullable=True)
    is_active_today     = Column(Boolean, default=False)

    user = relationship("User", back_populates="streak")
