"""Streak service for managing user streaks."""
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.streak import Streak
from app.models.user import User
from datetime import datetime, timedelta


class StreakService:
    """Streak management service."""

    @staticmethod
    async def get_or_create_streak(
        db: AsyncSession,
        user_id: int
    ) -> Streak:
        """Get or create a streak record for user."""
        stmt = select(Streak).where(Streak.user_id == user_id)
        result = await db.execute(stmt)
        streak = result.scalars().first()

        if not streak:
            streak = Streak(user_id=user_id)
            db.add(streak)
            await db.commit()
            await db.refresh(streak)

        return streak

    @staticmethod
    async def update_streak(
        db: AsyncSession,
        user_id: int
    ) -> Streak:
        """Update streak for user activity."""
        streak = await StreakService.get_or_create_streak(db, user_id)

        today = datetime.utcnow().date()
        last_activity = streak.last_activity_date.date() if streak.last_activity_date else None

        if last_activity == today:
            # Already active today
            return streak

        if last_activity == today - timedelta(days=1):
            # Streak continues
            streak.current_streak += 1
            if streak.current_streak > streak.longest_streak:
                streak.longest_streak = streak.current_streak
        else:
            # Streak broken or new
            streak.current_streak = 1
            streak.longest_streak = max(1, streak.longest_streak)

        streak.last_activity_date = datetime.utcnow()
        streak.is_active_today = True

        await db.commit()
        await db.refresh(streak)
        return streak

    @staticmethod
    async def reset_daily_activity(
        db: AsyncSession,
        user_id: int
    ) -> Streak:
        """Reset daily activity flag."""
        streak = await StreakService.get_or_create_streak(db, user_id)
        streak.is_active_today = False
        await db.commit()
        await db.refresh(streak)
        return streak

    @staticmethod
    async def get_user_streak(
        db: AsyncSession,
        user_id: int
    ) -> dict:
        """Get user streak information."""
        streak = await StreakService.get_or_create_streak(db, user_id)
        return {
            "current_streak": streak.current_streak,
            "longest_streak": streak.longest_streak,
            "is_active_today": streak.is_active_today,
            "last_activity_date": streak.last_activity_date
        }
