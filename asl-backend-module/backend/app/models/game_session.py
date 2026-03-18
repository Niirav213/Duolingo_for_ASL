"""GameSession SQLAlchemy model."""
from sqlalchemy import Column, Integer, Float, Boolean, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from app.db.base import Base


class GameSession(Base):
    __tablename__ = "game_sessions"

    id               = Column(Integer, primary_key=True, index=True)
    user_id          = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    lesson_id        = Column(Integer, nullable=False)
    score            = Column(Integer, default=0)
    accuracy         = Column(Float, default=0.0)
    duration_seconds = Column(Integer, default=0)
    completed        = Column(Boolean, default=False)
    created_at       = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="game_sessions")
