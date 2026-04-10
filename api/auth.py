"""
API key authentication dependency for the MedPredict API.
"""

import os

from fastapi import Header, HTTPException

from config import get_logger

logger = get_logger(__name__)

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    logger.warning("API_KEY not set! Using default (INSECURE for production)")
    API_KEY = "dev-key-12345"


async def verify_api_key(
    x_api_key: str = Header(..., description="API key for authentication"),
) -> str:
    """
    FastAPI dependency that validates the X-API-Key request header.

    Usage:
        curl -H "X-API-Key: your-key-here" https://api.example.com/query
    """
    if x_api_key != API_KEY:
        logger.warning("Invalid API key attempted: %s...", x_api_key[:8])
        raise HTTPException(
            status_code=401,
            detail="Invalid API key. Include 'X-API-Key' header with valid key.",
        )
    return x_api_key
