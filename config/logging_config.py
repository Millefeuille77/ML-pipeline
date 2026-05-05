"""Structured JSON logging with correlation IDs.

A `correlation_id` ContextVar is set per-request by the API middleware and
automatically attached to every log record emitted within that request scope.
"""
from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

CORRELATION_ID_VAR: ContextVar[str] = ContextVar("correlation_id", default="-")

_RESERVED_LOG_RECORD_ATTRS = frozenset(
    {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "message", "asctime",
    }
)


class JsonFormatter(logging.Formatter):
    """Format LogRecords as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        """Render a LogRecord as a JSON string.

        Args:
            record: The LogRecord to render.

        Returns:
            JSON-encoded string with a stable schema.
        """
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": CORRELATION_ID_VAR.get(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key in _RESERVED_LOG_RECORD_ATTRS or key.startswith("_"):
                continue
            payload[key] = value
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging with a single JSON stdout handler.

    Args:
        level: Minimum log level (DEBUG / INFO / WARNING / ERROR / CRITICAL).
    """
    root = logging.getLogger()
    root.setLevel(level.upper())
    for existing in list(root.handlers):
        root.removeHandler(existing)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonFormatter())
    handler.setLevel(level.upper())
    root.addHandler(handler)
    logging.getLogger("uvicorn.access").propagate = False


def set_correlation_id(correlation_id: str) -> None:
    """Set the correlation ID for the current async/thread context."""
    CORRELATION_ID_VAR.set(correlation_id)


def get_correlation_id() -> str:
    """Return the correlation ID for the current context (or '-' if unset)."""
    return CORRELATION_ID_VAR.get()
