"""Liveness/readiness route."""
from __future__ import annotations

import time
from typing import Final

from fastapi import APIRouter
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from config.settings import get_settings
from src.api.schemas import HealthResponse
from src.database.connection import check_health, get_engine
from src.models.registry import list_models
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/health", tags=["health"])

_TABLES_TO_COUNT: Final[tuple[str, ...]] = (
    "products", "daily_sales", "weekly_features", "enrichment_features",
    "demand_forecasts", "batch_predictions",
)
_PROCESS_START_MONOTONIC: float = time.monotonic()
# Frozenset mirrors _TABLES_TO_COUNT — only names in both are ever interpolated.
_ALLOWED_COUNT_TABLES: Final[frozenset[str]] = frozenset(_TABLES_TO_COUNT)


def _safe_row_counts() -> dict[str, int]:
    """Return COUNT(*) per table; names are allow-listed before interpolation."""
    counts: dict[str, int] = {table: 0 for table in _TABLES_TO_COUNT}
    engine = get_engine()
    with engine.connect() as connection:
        for table in _TABLES_TO_COUNT:
            if table not in _ALLOWED_COUNT_TABLES:
                continue  # defensive guard; unreachable in practice
            try:
                statement = text(f"SELECT COUNT(*) FROM {table}")
                counts[table] = int(connection.execute(statement).scalar() or 0)
            except SQLAlchemyError:
                logger.exception("health_count_failed", extra={"table": table})
    return counts


@router.get("", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return DB/model status, table row counts, and uptime.

    Returns:
        HealthResponse with current platform health.
    """
    db_ok = check_health()
    row_counts = _safe_row_counts() if db_ok else {table: 0 for table in _TABLES_TO_COUNT}
    models = list_models()
    model_status = "ok" if models else "missing"
    settings = get_settings()
    uptime_seconds = max(0.0, time.monotonic() - _PROCESS_START_MONOTONIC)
    return HealthResponse(
        db_status="ok" if db_ok else "down",
        model_status=model_status,
        uptime_seconds=uptime_seconds,
        version=settings.app_version,
        row_counts=row_counts,
    )
