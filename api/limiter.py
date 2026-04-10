"""
Shared rate limiter instance.

Defined here so both routes.py (decorators) and main.py (app state)
can import the same object without circular imports.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

QUERY_RATE_LIMIT = "20/hour"
STATS_RATE_LIMIT = "10/hour"
