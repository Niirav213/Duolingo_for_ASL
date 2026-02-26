"""
api/router.py
-------------
FastAPI routes exposed to the team's backend.
The pipeline is loaded ONCE at startup via lifespan.

Endpoints:
    POST /analyze   — analyze a single frame
    GET  /health    — check if service is running
    GET  /signs     — list all supported signs
"""

import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import AnalyzeFrameRequest, AnalyzeFrameResponse, HealthResponse
from api.pipeline import ASLPipeline

# Global pipeline instance — loaded once, reused forever
pipeline: ASLPipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load pipeline on startup, release on shutdown."""
    global pipeline
    print("[Server] Loading ASL pipeline...")
    pipeline = ASLPipeline(load_static=True, load_dynamic=True)
    print("[Server] Pipeline ready. Accepting requests.")
    yield
    # Shutdown
    if pipeline:
        pipeline.release()
    print("[Server] Shutdown complete.")


app = FastAPI(
    title="ASL CV Module",
    description="Computer vision pipeline for ASL sign recognition and feedback.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/analyze", response_model=AnalyzeFrameResponse)
async def analyze_frame(request: AnalyzeFrameRequest):
    """
    Analyze a single webcam frame for ASL sign detection and scoring.

    Accepts base64-encoded image, returns detection + feedback JSON.
    """
    try:
        result = pipeline.analyze_frame(
            frame_base64=request.frame_base64,
            target_sign=request.target_sign,
            mode=request.mode,
            include_landmarks=request.include_landmarks,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the CV service and models are loaded."""
    return HealthResponse(
        status="ok",
        model_loaded=pipeline is not None,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


@app.get("/signs")
async def list_signs():
    """Returns all supported ASL signs."""
    return {
        "static": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        "dynamic": [],  # populate when dynamic model is trained
    }
