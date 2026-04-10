"""
API route handlers for the MedPredict API.
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from auth import verify_api_key
from config import (
    ANTHROPIC_API_KEY,
    API_VERSION,
    CLAUDE_INPUT_TOKEN_COST,
    CLAUDE_OUTPUT_TOKEN_COST,
    get_logger,
)
from generation.generator import Generator
from graph.connection import get_driver
from limiter import HEALTH_RATE_LIMIT, QUERY_RATE_LIMIT, STATS_RATE_LIMIT, limiter
from models import DeviceInfo, HealthResponse, QueryRequest, QueryResponse
from retrieval.retriever import retrieve

logger = get_logger(__name__)

router = APIRouter()
generator = Generator()


# ============================================================================
# ENDPOINTS
# ============================================================================


@router.get("/", include_in_schema=False)
async def root():
    return {"message": "MedPredict API", "docs": "/docs", "health": "/health"}


@router.get("/health", response_model=HealthResponse)
@limiter.limit(HEALTH_RATE_LIMIT)
async def health_check(request: Request):
    """Health check endpoint (no auth required)."""
    neo4j_status = "connected"
    try:
        get_driver().verify_connectivity()
    except Exception:
        neo4j_status = "unavailable"

    generator_status = "ok" if ANTHROPIC_API_KEY else "api_key_missing"

    overall = "healthy" if neo4j_status == "connected" and generator_status == "ok" else "degraded"

    return HealthResponse(
        status=overall,
        version=API_VERSION,
        timestamp=datetime.now(timezone.utc).isoformat(),
        components={
            "neo4j": neo4j_status,
            "generator": generator_status,
        },
    )


@router.post("/query", response_model=QueryResponse)
@limiter.limit(QUERY_RATE_LIMIT)
async def query_devices(
    request: Request,
    query_request: QueryRequest,
    api_key: str = Depends(verify_api_key),
):
    """
    Query FDA 510(k) devices using natural language.

    **Authentication Required:** Include `X-API-Key` header with valid API key.

    **Rate Limit:** 20 queries per hour per IP address.

    **Example:**
```bash
    curl -X POST https://your-api.com/query \\
      -H "Content-Type: application/json" \\
      -H "X-API-Key: your-api-key-here" \\
      -d '{"query": "What devices are similar to Acuson?", "k": 5}'
```
    """
    logger.info("Query received: '%s' (k=%d)", query_request.query, query_request.k)

    try:
        logger.info("Retrieving similar devices...")
        subgraph = retrieve(query=query_request.query, top_k=query_request.k)
        retrieved_devices = subgraph["nodes"]
        logger.info("Retrieved %d devices", len(retrieved_devices))

        logger.info("Generating answer with Claude...")
        answer, tokens = generator.generate_with_usage(
            query=query_request.query,
            context=retrieved_devices,
        )

        cost = (tokens["input"] * CLAUDE_INPUT_TOKEN_COST) + (tokens["output"] * CLAUDE_OUTPUT_TOKEN_COST)
        logger.info(
            "Claude API used: input=%d output=%d cost=$%.4f",
            tokens["input"],
            tokens["output"],
            cost,
        )

        sources = [
            DeviceInfo(
                k_number=device.get("k_number", "Unknown"),
                device_name=device.get("device_name", "Unknown"),
                applicant=device.get("applicant", "Unknown"),
                decision_date=device.get("decision_date"),
                similarity_score=device.get("similarity_score"),
            )
            for device in retrieved_devices
        ]

        return QueryResponse(
            query=query_request.query,
            answer=answer,
            sources=sources,
            timestamp=datetime.now(timezone.utc).isoformat(),
            tokens_used=tokens,
        )

    except Exception as e:
        logger.error("Query failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while processing your query.")


@router.get("/stats")
@limiter.limit(STATS_RATE_LIMIT)
async def api_stats(
    request: Request,
    api_key: str = Depends(verify_api_key),
):
    """Get API usage statistics (requires authentication)."""
    return {
        "message": "Statistics endpoint",
        "note": "Implement analytics tracking for production",
    }


# ============================================================================
# ERROR HANDLERS
# ============================================================================


async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "detail": "Rate limit exceeded for this endpoint. Please try again later.",
            "retry_after": "1 hour",
        },
    )
