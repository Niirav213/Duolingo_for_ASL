"""Game API endpoints."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.schemas.lesson import GameSessionStart, GameSessionResponse
from app.models.user import User
from app.services.game_engine import GameEngine
from app.services.session_service import SessionService
from app.services.streak_service import StreakService
from app.db.session import get_db
from app.core.deps import get_current_user

router = APIRouter(prefix="/api/v1/game", tags=["game"])


@router.post("/session/start", response_model=GameSessionResponse)
async def start_game_session(
    session_data: GameSessionStart,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Start a new game session."""
    session = await GameEngine.start_game_session(
        db,
        current_user.id,
        session_data.lesson_id
    )

    return session


@router.post("/session/{session_id}/end")
async def end_game_session(
    session_id: int,
    score: int,
    accuracy: float,
    duration_seconds: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """End a game session."""
    session = await SessionService.get_session_by_id(db, session_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    updated_session = await GameEngine.end_game_session(
        db,
        session_id,
        score,
        accuracy,
        duration_seconds
    )

    # Calculate and record XP
    xp_earned = SessionService.calculate_session_xp(score, accuracy, duration_seconds)
    await GameEngine.record_progress(
        db,
        current_user.id,
        session.lesson_id,
        xp_earned
    )

    # Update streak
    await StreakService.update_streak(db, current_user.id)

    return {
        "session": updated_session,
        "xp_earned": xp_earned
    }


@router.get("/sessions", response_model=list[GameSessionResponse])
async def get_user_sessions(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    limit: int = 20
):
    """Get user's recent game sessions."""
    sessions = await SessionService.get_user_sessions(db, current_user.id, limit)
    return sessions


@router.get("/stats")
async def get_user_stats(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get user game statistics."""
    stats = await GameEngine.get_user_stats(db, current_user.id)
    return stats


@router.get("/streak")
async def get_user_streak(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get user streak information."""
    streak_info = await StreakService.get_user_streak(db, current_user.id)
    return streak_info
