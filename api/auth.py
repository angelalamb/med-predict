"""
API key authentication dependency for the MedPredict API.
"""

import os
import secrets

from fastapi import Header, HTTPException

from config import get_logger

logger = get_logger(__name__)

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError(
        "API_KEY environment variable is not set. "
        "Add it to your .env file before starting the server."
    )


async def verify_api_key(
    x_api_key: str = Header(..., description="API key for authentication"),
) -> str:
    """
    FastAPI dependency that validates the X-API-Key request header.

    Usage:
        curl -H "X-API-Key: your-key-here" https://api.example.com/query
    """
    if not secrets.compare_digest(x_api_key, API_KEY):
        logger.warning("Invalid API key attempted")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key. Include 'X-API-Key' header with valid key.",
        )
    return x_api_key
