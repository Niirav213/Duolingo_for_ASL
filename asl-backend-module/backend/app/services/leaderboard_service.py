"""Leaderboard service."""
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from app.models.user_progress import UserProgress
from app.models.streak import Streak


class LeaderboardService:
    """Leaderboard management service."""

    @staticmethod
    async def get_top_users_by_xp(
        db: AsyncSession,
        limit: int = 10
    ) -> list:
        """Get top users by total XP."""
        # This would typically be a more sophisticated query
        # For now, returning a mock implementation
        stmt = select(UserProgress).order_by(desc(UserProgress.total_xp)).limit(limit * 10)
        result = await db.execute(stmt)
        progress_records = result.scalars().all()

        user_xp_map = {}
        for record in progress_records:
            if record.user_id not in user_xp_map:
                user_xp_map[record.user_id] = 0
            user_xp_map[record.user_id] += record.xp_gained

        sorted_users = sorted(user_xp_map.items(), key=lambda x: x[1], reverse=True)
        return sorted_users[:limit]

    @staticmethod
    async def get_top_users_by_streak(
        db: AsyncSession,
        limit: int = 10
    ) -> list:
        """Get top users by current streak."""
        stmt = select(Streak).order_by(desc(Streak.current_streak)).limit(limit)
        result = await db.execute(stmt)
        streaks = result.scalars().all()

        return [
            {
                "user_id": s.user_id,
                "current_streak": s.current_streak,
                "longest_streak": s.longest_streak
            }
            for s in streaks
        ]

    @staticmethod
    async def get_user_rank_by_xp(
        db: AsyncSession,
        user_id: int
    ) -> dict:
        """Get user's rank by XP."""
        # Get user's total XP
        stmt = select(UserProgress).where(UserProgress.user_id == user_id)
        result = await db.execute(stmt)
        progress_records = result.scalars().all()
        user_xp = sum(p.xp_gained for p in progress_records)

        # Get all users count
        all_users_stmt = select(UserProgress.user_id).distinct()
        all_users_result = await db.execute(all_users_stmt)
        all_users = set(all_users_result.scalars().all())

        return {
            "user_id": user_id,
            "total_xp": user_xp,
            "total_users": len(all_users),
            "rank": 1  # Simplified, would need aggregation in real scenario
        }
