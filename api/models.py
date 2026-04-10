"""
Pydantic request and response schemas for the MedPredict API.
"""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class QueryRequest(BaseModel):
    """Request schema for device queries."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "What ultrasound devices are similar to Acuson S2000?",
                "k": 5,
            }
        }
    )

    query: str = Field(
        ...,
        description="Natural language query about FDA 510(k) devices",
        min_length=3,
        max_length=500,
    )
    k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of similar devices to retrieve",
    )


class DeviceInfo(BaseModel):
    """Device information schema."""

    k_number: str
    device_name: str
    applicant: str
    decision_date: Optional[str] = None
    similarity_score: Optional[float] = None


class QueryResponse(BaseModel):
    """Response schema for device queries."""

    query: str
    answer: str
    sources: List[DeviceInfo]
    timestamp: str
    tokens_used: Optional[dict] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    timestamp: str
    components: dict
