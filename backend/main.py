import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import get_settings
from backend.api.routes import papers, code, analysis, audit
from backend.services.logging import logger, setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    # Initialize logging
    setup_logging()
    logger.info("Starting Valid8 application")

    os.makedirs(settings.upload_dir, exist_ok=True)
    os.makedirs(settings.clone_dir, exist_ok=True)
    os.makedirs(settings.chroma_persist_dir, exist_ok=True)

    logger.info(f"AI Provider: {settings.ai_provider}")
    logger.info(f"CORS Origins: {settings.cors_origins}")

    yield

    logger.info("Shutting down Valid8 application")


settings = get_settings()

app = FastAPI(
    title="Valid8",
    description="Research Paper & Code Alignment Checker",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(papers.router, prefix="/api/papers", tags=["papers"])
app.include_router(code.router, prefix="/api/code", tags=["code"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
app.include_router(audit.router, prefix="/api/audit", tags=["audit"])


@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/api/status")
async def get_status():
    from backend.services.state import app_state
    return {
        "paper_loaded": app_state.paper is not None,
        "paper_name": app_state.paper.name if app_state.paper else None,
        "codebase_loaded": app_state.codebase is not None,
        "codebase_path": app_state.codebase.path if app_state.codebase else None,
        "indexed": app_state.is_indexed,
    }


# Serve frontend static files - MUST be last since it's a catch-all
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
