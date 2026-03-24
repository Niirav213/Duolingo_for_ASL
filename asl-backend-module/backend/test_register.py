import asyncio
from app.db.session import async_session
from app.services.auth_service import AuthService
from app.db.base import Base
from app.db.session import engine

async def run():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with async_session() as db:
        try:
            # check if user exists first to avoid unique constraint if we run multiple times
            from sqlalchemy import select
            from app.models.user import User
            stmt = select(User).where(User.username == "testuser125")
            res = await db.execute(stmt)
            if res.scalars().first():
                print("User already exists")
                return

            user = await AuthService.register_user(db, "testuser125", "test12@example.com", "password123")
            print("User created:", user.id)
            tokens = AuthService.create_tokens(user.id)
            print("Tokens:", tokens)
        except Exception as e:
            print("Error:", repr(e))

if __name__ == "__main__":
    asyncio.run(run())
