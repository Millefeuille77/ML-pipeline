"""Outermost error-handling middleware: structured JSON, no leaked stack traces."""
from __future__ import annotations

from typing import Final

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from src.utils.logger import get_logger

logger = get_logger(__name__)

_REQUEST_ID_HEADER: Final[str] = "X-Request-ID"


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Catch-all error boundary that converts unhandled exceptions to JSON 500s."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Run the downstream handler; on exception, emit a sanitized JSON 500."""
        request_id = getattr(request.state, "request_id", "-")
        try:
            return await call_next(request)
        except Exception:  # noqa: BLE001 — intentional outer boundary
            logger.exception(
                "unhandled_request_exception",
                extra={"path": request.url.path, "request_id": request_id},
            )
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "internal_error",
                    "detail": "An unexpected error occurred. Please retry shortly.",
                    "request_id": request_id,
                },
                headers={_REQUEST_ID_HEADER: request_id},
            )
