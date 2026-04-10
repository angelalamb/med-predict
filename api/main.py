"""
MedPredict API entry point.

Creates the FastAPI app, registers middleware, and wires up routes.
Run with: python3 api/main.py
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Add project root to path so retrieval/generation/config are importable
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded

from config import API_VERSION, get_logger, validate
from limiter import limiter
from routes import rate_limit_handler, router

logger = get_logger(__name__)


# ============================================================================
# APP
# ============================================================================


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("=" * 70)
    logger.info("MedPredict API starting up")
    logger.info("=" * 70)
    validate()
    logger.info("✓ Configuration validated")
    logger.info("✓ MedPredict API is ready!")
    logger.info("=" * 70)
    yield
    logger.info("MedPredict API shutting down")


app = FastAPI(
    title="MedPredict API",
    description="FDA 510(k) Device Similarity Search and RAG System",
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    dev = os.getenv("ENVIRONMENT") == "development"
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=dev, log_level="info")
