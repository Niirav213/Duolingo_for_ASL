"""Setup script to create demo data and initialize the database."""
import asyncio
from app.db.session import engine, async_session
from app.db.base import Base
from app.models.user import User, Streak
from app.core.security import hash_password


async def setup():
    """Create tables and demo data."""
    print("Setting up database...")

    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✓ Database tables created")

    # Create demo user
    async with async_session() as session:
        from sqlalchemy import select

        # Check if demo user exists
        stmt = select(User).where(User.username == "demo")
        result = await session.execute(stmt)
        exists = result.scalars().first()

        if not exists:
            demo_user = User(
                username="demo",
                email="demo@example.com",
                hashed_password=hash_password("demo123"),
                is_active=True
            )
            session.add(demo_user)
            await session.commit()
            await session.refresh(demo_user)

            # Create streak record
            streak = Streak(user_id=demo_user.id, current_streak=0, longest_streak=0)
            session.add(streak)
            await session.commit()

            print("✓ Demo user created (username: demo, password: demo123)")
        else:
            print("✓ Demo user already exists")

    print("✓ Setup complete!")


if __name__ == "__main__":
    asyncio.run(setup())
