"""
api/schemas.py
--------------
Pydantic models defining the JSON contract between
this CV module and the rest of the team's backend/frontend.

This is the MOST IMPORTANT file for team collaboration.
Any change here must be communicated to the team.
"""

from pydantic import BaseModel
from typing import Optional


# ─────────────────────────────────────────────
# Request
# ─────────────────────────────────────────────

class AnalyzeFrameRequest(BaseModel):
    """
    Sent by the frontend/backend to this CV module.
    Frame should be base64-encoded JPEG/PNG.
    """
    frame_base64: str               # base64 encoded image
    target_sign: str                # e.g. "A", "B", "HELLO"
    mode: str = "static"            # "static" or "dynamic"
    include_landmarks: bool = False # whether to return raw landmark data


# ─────────────────────────────────────────────
# Response
# ─────────────────────────────────────────────

class JointScores(BaseModel):
    thumb_mcp: float = 0.0
    thumb_ip: float = 0.0
    index_mcp: float = 0.0
    index_pip: float = 0.0
    middle_mcp: float = 0.0
    middle_pip: float = 0.0
    ring_mcp: float = 0.0
    ring_pip: float = 0.0
    pinky_mcp: float = 0.0
    pinky_pip: float = 0.0
    position: float = 0.0
    orientation: float = 0.0


class LandmarkPoint(BaseModel):
    x: float
    y: float
    z: float


class AnalyzeFrameResponse(BaseModel):
    """
    Returned by this CV module for every frame analyzed.
    Frontend uses this to render feedback UI.
    """
    # ── Detection ──
    hand_detected: bool
    detected_sign: str              # what the model thinks the sign is
    confidence: float               # 0.0 to 1.0

    # ── Scoring ──
    overall_score: float            # 0 to 100
    is_correct: bool                # True if score >= threshold
    joint_scores: JointScores

    # ── Feedback ──
    messages: list[str]             # e.g. ["Curl your index finger more"]
    praise: str                     # e.g. "Great job!"
    emoji: str                      # e.g. "✅"

    # ── Visual ──
    joint_colors: dict[str, str]    # e.g. {"index_pip": "red", "thumb_mcp": "green"}

    # ── Optional raw data ──
    landmarks: Optional[dict] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
