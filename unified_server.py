import sys
import os
from pathlib import Path
import asyncio

# --- PATH SETUP ---
# Absolute project root
root_dir = Path(__file__).parent.absolute()
backend_dir = root_dir / "asl-backend-module" / "backend"
cv_dir = root_dir / "asl-cv-module"

# Add directories to sys.path so we can import 'app' and 'api'
sys.path.append(str(backend_dir))
sys.path.append(str(cv_dir))

# --- FASTAPI IMPORTS ---
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# --- MODULE IMPORTS ---
try:
    from app.db.base import Base
    from app.db.session import engine
    from app.api import auth, game, ws_game
    from app.core.config import settings
    
    from api.router import app as cv_app
    from api.pipeline import ASLPipeline
    import api.router as cv_router
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# --- UNIFIED LIFESPAN ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Backend Startup: Create database tables
    print("[Unified] Initializing database...")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    except Exception as e:
        print(f"[Unified] Database error: {e}")
    
    # 2. CV Module Startup: Load pipeline with correct working directory
    print("[Unified] Loading ASL pipeline...")
    old_cwd = os.getcwd()
    os.chdir(str(cv_dir))
    try:
        cv_router.pipeline = ASLPipeline(load_static=True, load_dynamic=True)
    except Exception as e:
        print(f"[Unified] Pipeline loading error: {e}")
    finally:
        os.chdir(old_cwd)
    print("[Unified] Pipeline ready.")
    
    yield
    
    # --- SHUTDOWN ---
    print("[Unified] Shutting down...")
    try:
        await engine.dispose()
    except:
        pass
    if cv_router.pipeline:
        try:
            cv_router.pipeline.release()
        except:
            pass

# --- APP INITIALIZATION ---
app = FastAPI(
    title="Unified ASL Learning Platform",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware (combining settings)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API ROUTES ---
# Include routers from backend
app.include_router(auth.router)
app.include_router(game.router)
app.include_router(ws_game.router)

# Mount CV Module at /cv
app.mount("/cv", cv_app)

# --- FRONTEND SERVING ---
dist_path = root_dir / "ASL_Frontend" / "dist"

# Health Check
@app.get("/health_unified")
async def health_unified():
    return {
        "status": "ok",
        "backend": "online",
        "cv": "online" if cv_router.pipeline else "error",
        "frontend": "built" if dist_path.exists() else "not found"
    }

# Catch-all for Frontend Files and SPA Routing
@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    # Skip API/CV paths (FastAPI should have caught them but this is a fallback)
    if full_path.startswith(("api/", "cv/", "docs", "redoc", "openapi.json")):
         return JSONResponse(status_code=404, content={"detail": "Not Found"})

    # Check if file exists in dist
    file_path = dist_path / full_path
    if full_path and file_path.is_file():
        return FileResponse(file_path)
    
    # Static fallback for CSS/JS in assets
    if full_path.startswith("assets/"):
        return FileResponse(dist_path / full_path)

    # Fallback to index.html for SPA routing
    index_file = dist_path / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    
    return JSONResponse(status_code=404, content={"detail": "Frontend build not found at " + str(index_file)})

if __name__ == "__main__":
    import uvicorn
    # Change to root dir for consistent relative paths if needed
    os.chdir(str(root_dir))
    uvicorn.run(app, host="0.0.0.0", port=8000)
