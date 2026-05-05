"""In-memory sliding-window rate limiter (no Redis)."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from threading import Lock
from typing import Deque, Final

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

_WINDOW_SECONDS: Final[float] = 60.0
_API_KEY_HEADER: Final[str] = "X-API-Key"
_REQUEST_ID_HEADER: Final[str] = "X-Request-ID"


def _bucket_key(api_key: str | None, request: Request) -> str:
    """Build a distinct bucket key from API key or client IP.

    Uses `X-Forwarded-For` first hop when present (trust only one hop).
    Prefixes `key:` vs `ip:` to keep key-based and IP-based buckets apart.

    Args:
        api_key: Value of the `X-API-Key` header, or None.
        request: Incoming ASGI request.

    Returns:
        A non-empty string suitable as a rate-limiter bucket key.
    """
    if api_key:
        return f"key:{api_key}"
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        client_ip = forwarded_for.split(",", 1)[0].strip()
    else:
        client_ip = (request.client.host if request.client else "unknown")
    return f"ip:{client_ip}"


class SlidingWindowRateLimiter:
    """Per-key sliding window counter."""

    def __init__(self, max_requests: int, window_seconds: float = _WINDOW_SECONDS) -> None:
        """Create a new limiter.

        Args:
            max_requests: Allowed requests per window.
            window_seconds: Window length in seconds.
        """
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._buckets: dict[str, Deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        """Return True if a request from `key` should be allowed."""
        timestamp = time.monotonic() if now is None else now
        with self._lock:
            bucket = self._buckets[key]
            cutoff = timestamp - self._window_seconds
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= self._max_requests:
                return False
            bucket.append(timestamp)
            return True

    def reset(self) -> None:
        """Drop all per-key counters (for tests)."""
        with self._lock:
            self._buckets.clear()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """ASGI middleware enforcing per-API-key rate limits."""

    def __init__(self, app, limiter: SlidingWindowRateLimiter | None = None) -> None:
        """Initialize the middleware.

        Args:
            app: Wrapped ASGI app.
            limiter: Optional limiter; defaults to one sized from settings.
        """
        super().__init__(app)
        settings = get_settings()
        self._limiter = limiter or SlidingWindowRateLimiter(
            max_requests=settings.api_rate_limit_per_min
        )

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Allow or reject based on per-key sliding window state.

        When `X-API-Key` is absent (e.g. exempt paths), fall back to the
        client IP so auth-exempt routes are still throttled.  Bucket keys
        are prefixed with `key:` or `ip:` to keep the namespaces distinct.
        """
        api_key = request.headers.get(_API_KEY_HEADER)
        request_id = getattr(request.state, "request_id", "-")
        bucket_key = _bucket_key(api_key, request)
        if not self._limiter.allow(bucket_key):
            logger.warning(
                "rate_limit_exceeded",
                extra={"path": request.url.path, "request_id": request_id},
            )
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "rate_limited",
                    "detail": "Per-key request quota exceeded; retry after one minute.",
                    "request_id": request_id,
                },
                headers={_REQUEST_ID_HEADER: request_id},
            )
        return await call_next(request)
