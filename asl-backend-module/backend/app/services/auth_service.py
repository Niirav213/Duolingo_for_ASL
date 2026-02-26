"""Authentication service."""
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.user import User
from app.core.security import hash_password, verify_password, create_token_pair


class AuthService:
    """Authentication service."""

    @staticmethod
    async def register_user(
        db: AsyncSession,
        username: str,
        email: str,
        password: str
    ) -> User:
        """Register a new user."""
        hashed_password = hash_password(password)

        new_user = User(
            username=username,
            email=email,
            hashed_password=hashed_password
        )

        db.add(new_user)
        await db.commit()
        await db.refresh(new_user)
        return new_user

    @staticmethod
    async def authenticate_user(
        db: AsyncSession,
        username: str,
        password: str
    ) -> User:
        """Authenticate user and return user if valid."""
        stmt = select(User).where(User.username == username)
        result = await db.execute(stmt)
        user = result.scalars().first()

        if not user:
            return None

        if not verify_password(password, user.hashed_password):
            return None

        return user

    @staticmethod
    def create_tokens(user_id: int) -> dict:
        """Create access and refresh tokens for user."""
        return create_token_pair(user_id)
