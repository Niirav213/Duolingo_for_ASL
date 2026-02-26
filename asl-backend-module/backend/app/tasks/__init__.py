"""Celery tasks for async operations."""
from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "asl_platform",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)


@celery_app.task
def send_notification(user_id: int, message: str):
    """Send notification to user."""
    print(f"Sending notification to user {user_id}: {message}")
    return {"status": "sent", "user_id": user_id}


@celery_app.task
def update_leaderboard():
    """Update leaderboard standings."""
    print("Updating leaderboard...")
    return {"status": "updated"}


@celery_app.task
def calculate_daily_streaks():
    """Calculate daily streaks for all users."""
    print("Calculating daily streaks...")
    return {"status": "calculated"}


@celery_app.task
def generate_report(user_id: int):
    """Generate user progress report."""
    print(f"Generating report for user {user_id}")
    return {"status": "generated", "user_id": user_id}
