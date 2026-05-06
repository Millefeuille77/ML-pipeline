"""Liveness/readiness route."""
from __future__ import annotations

import time
from datetime import date, timedelta
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
    "prediction_log", "model_lineage", "model_performance_live",
)
_PROCESS_START_MONOTONIC: float = time.monotonic()
# Frozenset mirrors _TABLES_TO_COUNT — only names in both are ever interpolated.
_ALLOWED_COUNT_TABLES: Final[frozenset[str]] = frozenset(_TABLES_TO_COUNT)
_STALE_MODEL_DAYS: Final[int] = 14
_LIVE_MAPE_WINDOW_DAYS: Final[int] = 7


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


def _days_since_last_retrain() -> int | None:
    """Query model_lineage for the most recent completed_at timestamp."""
    try:
        engine = get_engine()
        with engine.connect() as connection:
            result = connection.execute(
                text("SELECT MAX(completed_at) FROM model_lineage WHERE completed_at IS NOT NULL")
            ).scalar()
        if result is None:
            return None
        delta = date.today() - result.date()
        return max(0, delta.days)
    except SQLAlchemyError:
        logger.exception("health_retrain_query_failed")
        return None


def _live_mape_by_category() -> dict[str, float]:
    """Return average live_mape per category from last 7 days of model_performance_live."""
    try:
        engine = get_engine()
        cutoff = date.today() - timedelta(days=_LIVE_MAPE_WINDOW_DAYS)
        with engine.connect() as connection:
            rows = connection.execute(
                text(
                    "SELECT category, AVG(live_mape) AS avg_mape "
                    "FROM model_performance_live "
                    "WHERE scored_date >= :cutoff "
                    "GROUP BY category"
                ),
                {"cutoff": cutoff},
            ).fetchall()
        return {str(row[0]): float(row[1]) for row in rows}
    except SQLAlchemyError:
        logger.exception("health_live_mape_query_failed")
        return {}


def _stale_model_warnings(days_since: int | None) -> list[str]:
    """Return category names whose newest model exceeds _STALE_MODEL_DAYS old."""
    if days_since is None:
        return []
    try:
        engine = get_engine()
        cutoff = date.today() - timedelta(days=_STALE_MODEL_DAYS)
        with engine.connect() as connection:
            rows = connection.execute(
                text(
                    "SELECT DISTINCT category FROM model_lineage "
                    "WHERE category NOT IN ("
                    "  SELECT category FROM model_lineage "
                    "  WHERE completed_at >= :cutoff AND completed_at IS NOT NULL"
                    ")"
                ),
                {"cutoff": cutoff},
            ).fetchall()
        return [str(row[0]) for row in rows]
    except SQLAlchemyError:
        logger.exception("health_stale_model_query_failed")
        return []


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
    days_since = _days_since_last_retrain() if db_ok else None
    live_mape = _live_mape_by_category() if db_ok else {}
    stale_warnings = _stale_model_warnings(days_since) if db_ok else []
    return HealthResponse(
        db_status="ok" if db_ok else "down",
        model_status=model_status,
        uptime_seconds=uptime_seconds,
        version=settings.app_version,
        row_counts=row_counts,
        days_since_last_retrain=days_since,
        live_mape_by_category=live_mape,
        stale_model_warning=stale_warnings,
    )
