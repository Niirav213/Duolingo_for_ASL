"""Test game engine service."""
import pytest
from app.services.game_engine import GameEngine
from app.models.user import User
from app.models.streak import Streak


@pytest.mark.asyncio
async def test_start_game_session(async_session):
    """Test starting a game session."""
    # Create user
    user = User(username="testuser", email="test@example.com", hashed_password="hashed")
    async_session.add(user)
    await async_session.commit()
    await async_session.refresh(user)

    # Start game session
    session = await GameEngine.start_game_session(async_session, user.id, lesson_id=1)

    assert session.user_id == user.id
    assert session.lesson_id == 1
    assert session.completed is False


@pytest.mark.asyncio
async def test_record_progress(async_session):
    """Test recording user progress."""
    # Create user
    user = User(username="testuser", email="test@example.com", hashed_password="hashed")
    async_session.add(user)
    await async_session.commit()
    await async_session.refresh(user)

    # Record progress
    progress = await GameEngine.record_progress(
        async_session,
        user.id,
        lesson_id=1,
        xp_gained=100
    )

    assert progress.user_id == user.id
    assert progress.lesson_id == 1
    assert progress.xp_gained == 100
