"""Session service for managing game sessions."""
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.game_session import GameSession
from app.models.user import User


class SessionService:
    """Game session management service."""

    @staticmethod
    async def get_user_sessions(
        db: AsyncSession,
        user_id: int,
        limit: int = 20
    ) -> list:
        """Get user's recent game sessions."""
        stmt = select(GameSession).where(
            GameSession.user_id == user_id
        ).order_by(GameSession.created_at.desc()).limit(limit)

        result = await db.execute(stmt)
        return result.scalars().all()

    @staticmethod
    async def get_session_by_id(
        db: AsyncSession,
        session_id: int
    ) -> GameSession:
        """Get a specific game session."""
        stmt = select(GameSession).where(GameSession.id == session_id)
        result = await db.execute(stmt)
        return result.scalars().first()

    @staticmethod
    async def get_session_stats(
        db: AsyncSession,
        session_id: int
    ) -> dict:
        """Get statistics for a specific session."""
        session = await SessionService.get_session_by_id(db, session_id)

        if not session:
            return {}

        return {
            "id": session.id,
            "user_id": session.user_id,
            "lesson_id": session.lesson_id,
            "score": session.score,
            "accuracy": session.accuracy,
            "duration_seconds": session.duration_seconds,
            "completed": session.completed,
            "created_at": session.created_at
        }

    @staticmethod
    async def calculate_session_xp(
        score: int,
        accuracy: float,
        duration_seconds: int
    ) -> int:
        """Calculate XP gained from a session."""
        base_xp = score
        accuracy_bonus = int(accuracy * 100)
        time_bonus = max(0, 300 - duration_seconds) // 10

        total_xp = base_xp + accuracy_bonus + time_bonus
        return max(10, total_xp)
