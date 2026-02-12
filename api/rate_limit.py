"""
Per-endpoint rate limiting for FastAPI.

Add to any route with Depends(rate_limit("endpoint_key", requests_per_minute=60)).
Redis-backed so limits are shared across API replicas. Pass 0 to disable.
"""

import time
from typing import Callable

from fastapi import HTTPException, status

from api.config import settings


def rate_limit(endpoint_key: str, requests_per_minute: int) -> Callable:
    """
    FastAPI dependency that enforces a rate limit for this endpoint.

    Usage:
        @router.get("/evidence")
        async def get_evidence(
            _: None = Depends(rate_limit("tee_evidence", 60)),
            ...
        ):
            ...
    """

    async def _rate_limit() -> None:
        if requests_per_minute <= 0:
            return
        window = int(time.time() // 60)
        key = f"rate_limit:{endpoint_key}:{window}"
        redis = settings.redis_client
        count = await redis.incr(key)
        if count == 1:
            await redis.expire(key, 120)
        if count > requests_per_minute:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Try again later.",
            )

    return _rate_limit
