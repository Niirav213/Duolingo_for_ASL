"""Game engine service."""
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.user_progress import UserProgress
from app.models.game_session import GameSession
from app.models.streak import Streak
from datetime import datetime


class GameEngine:
    """Game engine for managing game logic."""

    @staticmethod
    async def start_game_session(
        db: AsyncSession,
        user_id: int,
        lesson_id: int
    ) -> GameSession:
        """Start a new game session."""
        session = GameSession(
            user_id=user_id,
            lesson_id=lesson_id
        )
        db.add(session)
        await db.commit()
        await db.refresh(session)
        return session

    @staticmethod
    async def end_game_session(
        db: AsyncSession,
        session_id: int,
        score: int,
        accuracy: float,
        duration_seconds: int
    ) -> GameSession:
        """End a game session and record results."""
        stmt = select(GameSession).where(GameSession.id == session_id)
        result = await db.execute(stmt)
        session = result.scalars().first()

        if session:
            session.score = score
            session.accuracy = accuracy
            session.duration_seconds = duration_seconds
            session.completed = True
            await db.commit()
            await db.refresh(session)

        return session

    @staticmethod
    async def record_progress(
        db: AsyncSession,
        user_id: int,
        lesson_id: int,
        xp_gained: int
    ) -> UserProgress:
        """Record user progress for a lesson."""
        progress = UserProgress(
            user_id=user_id,
            lesson_id=lesson_id,
            xp_gained=xp_gained,
            total_xp=xp_gained
        )
        db.add(progress)
        await db.commit()
        await db.refresh(progress)
        return progress

    @staticmethod
    async def get_user_stats(
        db: AsyncSession,
        user_id: int
    ) -> dict:
        """Get user statistics."""
        # Get total XP
        stmt = select(UserProgress).where(UserProgress.user_id == user_id)
        result = await db.execute(stmt)
        progress_records = result.scalars().all()
        total_xp = sum(p.xp_gained for p in progress_records)

        # Get streak info
        streak_stmt = select(Streak).where(Streak.user_id == user_id)
        streak_result = await db.execute(streak_stmt)
        streak = streak_result.scalars().first()

        return {
            "total_xp": total_xp,
            "current_streak": streak.current_streak if streak else 0,
            "longest_streak": streak.longest_streak if streak else 0,
            "lessons_completed": len(progress_records)
        }
