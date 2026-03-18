"""UserProgress SQLAlchemy model."""
from sqlalchemy import Column, Integer, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from app.db.base import Base


class UserProgress(Base):
    __tablename__ = "user_progress"

    id            = Column(Integer, primary_key=True, index=True)
    user_id       = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    lesson_id     = Column(Integer, nullable=False, index=True)
    xp_gained     = Column(Integer, default=0)
    total_xp      = Column(Integer, default=0)
    current_level = Column(Integer, default=1)
    completed_at  = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="progress")
