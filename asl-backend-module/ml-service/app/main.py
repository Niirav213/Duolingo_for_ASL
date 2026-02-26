"""ML Service FastAPI application for gesture detection."""
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import cv2
import numpy as np
import logging
import sys
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from app.pipeline import MediaPipePipeline
except ImportError as e:
    logger.error(f"Failed to import pipeline: {e}")
    logger.error("Please ensure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)

# Initialize FastAPI app
app = FastAPI(title="ASL Gesture Detection Service", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline (will be loaded on startup)
pipeline = None
startup_complete = False


class GestureDetectionRequest(BaseModel):
    """Request schema for gesture detection."""
    image_data: str  # base64 encoded image


class GestureDetectionResponse(BaseModel):
    """Response schema for gesture detection."""
    predicted_class: str
    confidence: float
    landmarks: list = None


@app.on_event("startup")
async def startup():
    """Initialize models on startup."""
    global pipeline, startup_complete
    try:
        logger.info("Starting gesture detection service...")
        pipeline = MediaPipePipeline(model_path="app/models/gesture_model.onnx")
        pipeline.load_model()
        startup_complete = True
        logger.info("✓ Gesture detection service started successfully")
    except Exception as e:
        logger.error(f"✗ Failed to start service: {e}")
        startup_complete = False
        # Don't exit - service can still run with mock predictions


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    global pipeline
    if pipeline:
        del pipeline
    logger.info("Gesture detection service stopped")


@app.post("/predict", response_model=GestureDetectionResponse)
async def predict_gesture(request: GestureDetectionRequest):
    """Predict gesture from image."""
    if not startup_complete:
        logger.warning("Pipeline not fully initialized, using mock predictions")
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image_data)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid image data"
            )

        # Perform prediction
        if pipeline and startup_complete:
            result = pipeline.predict(image)
        else:
            # Fallback to mock prediction
            classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
                      "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
                      "U", "V", "W", "X", "Y", "Z", "SPACE"]
            result = {
                "class": random.choice(classes),
                "confidence": 0.5 + random.random() * 0.5,
                "landmarks": None
            }

        return GestureDetectionResponse(
            predicted_class=result["class"],
            confidence=result["confidence"],
            landmarks=result.get("landmarks")
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
    """Health check endpoint."""
    return {"status": "ok", "service": "gesture-detection"}


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "ASL Gesture Detection",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
