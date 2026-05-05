"""
Shared rate limiter instance.

Defined here so both routes.py (decorators) and main.py (app state)
can import the same object without circular imports.
"""

from fastapi import Request
from slowapi import Limiter


def _get_real_ip(request: Request) -> str:
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.client.host


limiter = Limiter(key_func=_get_real_ip)

QUERY_RATE_LIMIT = "20/hour"
STATS_RATE_LIMIT = "10/hour"
HEALTH_RATE_LIMIT = "60/minute"
