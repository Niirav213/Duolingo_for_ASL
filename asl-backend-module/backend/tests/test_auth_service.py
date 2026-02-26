"""Test authentication service."""
import pytest
from app.services.auth_service import AuthService
from app.models.user import User


@pytest.mark.asyncio
async def test_register_user(async_session):
    """Test user registration."""
    user = await AuthService.register_user(
        async_session,
        username="testuser",
        email="test@example.com",
        password="password123"
    )

    assert user.username == "testuser"
    assert user.email == "test@example.com"
    assert user.is_active is True
    assert user.id is not None


@pytest.mark.asyncio
async def test_authenticate_user(async_session):
    """Test user authentication."""
    # Create user
    user = await AuthService.register_user(
        async_session,
        username="testuser",
        email="test@example.com",
        password="password123"
    )

    # Authenticate with correct password
    authenticated_user = await AuthService.authenticate_user(
        async_session,
        username="testuser",
        password="password123"
    )

    assert authenticated_user is not None
    assert authenticated_user.id == user.id

    # Authenticate with wrong password
    authenticated_user = await AuthService.authenticate_user(
        async_session,
        username="testuser",
        password="wrongpassword"
    )

    assert authenticated_user is None


@pytest.mark.asyncio
async def test_create_tokens():
    """Test token creation."""
    tokens = AuthService.create_tokens(user_id=1)

    assert "access_token" in tokens
    assert "refresh_token" in tokens
    assert tokens["token_type"] == "bearer"
    assert len(tokens["access_token"]) > 0
    assert len(tokens["refresh_token"]) > 0
