"""
ml-service/app/main.py
-----------------------
FastAPI ML service — updated to use the real asl-cv-module pipeline
and return score + feedback fields alongside the prediction.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import base64
import cv2
import numpy as np
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from app.pipeline import MediaPipePipeline
except ImportError as e:
    logger.error(f"Failed to import pipeline: {e}")
    sys.exit(1)

app = FastAPI(title="ASL Gesture Detection Service", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = None
startup_complete = False


# ─────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────

class GestureDetectionRequest(BaseModel):
    """Request schema — same as before so backend needs no changes."""
    image_data: str          # base64 encoded image (no data-URL prefix needed)
    target_sign: str = ""    # optional: if provided, score + feedback are returned


class GestureDetectionResponse(BaseModel):
    """
    Extended response — backwards compatible.
    predicted_class and confidence are the same fields as before.
    score, is_correct, messages, joint_colors are new fields for the frontend.
    """
    predicted_class: str
    confidence: float
    landmarks: Optional[list] = None

    # ── New fields ──
    score: float = 0.0
    is_correct: bool = False
    messages: list = []
    joint_colors: dict = {}


# ─────────────────────────────────────────────
# Startup / Shutdown
# ─────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    global pipeline, startup_complete
    try:
        logger.info("Starting gesture detection service...")
        pipeline = MediaPipePipeline()
        pipeline.load_model()
        startup_complete = pipeline.model_loaded
        if startup_complete:
            logger.info("Gesture detection service started successfully")
        else:
            logger.warning("Service started but model not loaded — check checkpoint path")
    except Exception as e:
        logger.error(f"Failed to start service: {e}")
        startup_complete = False


@app.on_event("shutdown")
async def shutdown():
    global pipeline
    if pipeline:
        del pipeline
    logger.info("Gesture detection service stopped")


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.post("/predict", response_model=GestureDetectionResponse)
async def predict_gesture(request: GestureDetectionRequest):
    """Predict gesture from base64 image, with optional scoring."""
    try:
        # Decode image
        image_data = base64.b64decode(request.image_data)
        nparr  = np.frombuffer(image_data, np.uint8)
        image  = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid image data"
            )

        # Run pipeline
        result = pipeline.predict(image, target_sign=request.target_sign)

        return GestureDetectionResponse(
            predicted_class = result["class"],
            confidence      = result["confidence"],
            landmarks       = result.get("landmarks"),
            score           = result.get("score", 0.0),
            is_correct      = result.get("is_correct", False),
            messages        = result.get("messages", []),
            joint_colors    = result.get("joint_colors", {}),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/health")
async def health_check():
    return {
        "status":        "ok",
        "service":       "gesture-detection",
        "model_loaded":  startup_complete,
    }


@app.get("/")
async def root():
    return {
        "service":   "ASL Gesture Detection",
        "version":   "2.0.0",
        "endpoints": {
            "predict": "POST /predict",
            "health":  "GET  /health",
            "docs":    "GET  /docs",
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)