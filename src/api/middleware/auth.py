"""API key authentication middleware."""
from __future__ import annotations

from typing import Final
from uuid import uuid4

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from config.logging_config import set_correlation_id
from config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

_API_KEY_HEADER: Final[str] = "X-API-Key"
_REQUEST_ID_HEADER: Final[str] = "X-Request-ID"
_EXEMPT_PATHS: Final[frozenset[str]] = frozenset(
    {"/api/v1/health", "/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"}
)


class ApiKeyAuthMiddleware(BaseHTTPMiddleware):
    """Validate `X-API-Key` against `settings.api_key`.

    Public paths (`/api/v1/health`, `/docs`, `/redoc`, `/openapi.json`,
    `/docs/oauth2-redirect`) bypass the check via strict set membership.
    Every request is tagged with a correlation ID propagated via
    response header `X-Request-ID`.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Validate the API key (when required) and forward the request."""
        request_id = request.headers.get(_REQUEST_ID_HEADER) or uuid4().hex
        set_correlation_id(request_id)
        request.state.request_id = request_id
        if _is_exempt(request.url.path):
            response = await call_next(request)
            response.headers[_REQUEST_ID_HEADER] = request_id
            return response
        provided = request.headers.get(_API_KEY_HEADER)
        expected = get_settings().api_key.get_secret_value()
        if not provided or not _safe_compare(provided, expected):
            logger.warning(
                "auth_rejected",
                extra={"path": request.url.path, "has_key": bool(provided)},
            )
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": "unauthorized",
                    "detail": "Missing or invalid X-API-Key header.",
                    "request_id": request_id,
                },
                headers={_REQUEST_ID_HEADER: request_id},
            )
        response = await call_next(request)
        response.headers[_REQUEST_ID_HEADER] = request_id
        return response


def _is_exempt(path: str) -> bool:
    """Return True if `path` does not require authentication.

    Uses strict set membership — no prefix matching — to prevent paths like
    `/docsattack` from being inadvertently whitelisted.
    """
    return path in _EXEMPT_PATHS


def _safe_compare(provided: str, expected: str) -> bool:
    """Constant-time comparison guard against timing attacks."""
    if len(provided) != len(expected):
        return False
    accumulator = 0
    for left, right in zip(provided, expected):
        accumulator |= ord(left) ^ ord(right)
    return accumulator == 0
