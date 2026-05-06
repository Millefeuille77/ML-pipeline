"""Loaders: idempotent batched upserts via parameterized SQL."""
from __future__ import annotations

import re
from typing import Final

import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from src.database.connection import session_scope
from src.utils.logger import get_logger

logger = get_logger(__name__)

_BATCH_SIZE: Final[int] = 5000
_IDENTIFIER_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z_][a-z0-9_]*$")

_ALLOWED_TABLES: Final[frozenset[str]] = frozenset(
    {
        "products",
        "daily_sales",
        "weekly_features",
        "enrichment_features",
        "demand_forecasts",
        "batch_predictions",
        "prediction_log",
        "model_lineage",
        "model_performance_live",
    }
)


def _validate_identifier(name: str) -> str:
    """Validate a SQL identifier against a safe pattern and raise on mismatch.

    Args:
        name: Identifier string (table name or column name).

    Returns:
        The unchanged `name` if valid.

    Raises:
        ValueError: If `name` contains characters outside `[a-z0-9_]`.
    """
    if not _IDENTIFIER_RE.fullmatch(name):
        raise ValueError(f"unsafe SQL identifier: {name!r}")
    return name


def _validate_table_name(table: str) -> str:
    """Reject any table name not in the allow-list (defense in depth)."""
    if table not in _ALLOWED_TABLES:
        raise ValueError(f"refusing to load into unknown table: {table!r}")
    return _validate_identifier(table)


def upsert_dataframe(
    table: str,
    frame: pd.DataFrame,
    conflict_cols: list[str],
    batch_size: int = _BATCH_SIZE,
) -> int:
    """Upsert a DataFrame into `table` using ON CONFLICT DO UPDATE.

    Args:
        table: Allow-listed target table name.
        frame: DataFrame whose columns map 1:1 to table columns.
        conflict_cols: Columns forming the unique conflict target.
        batch_size: Rows per transaction batch.

    Returns:
        Total rows upserted.

    Raises:
        ValueError: If `table` is not allow-listed or `frame` is malformed.
    """
    _validate_table_name(table)
    if frame.empty:
        return 0
    columns = [_validate_identifier(col) for col in frame.columns]
    for col in conflict_cols:
        _validate_identifier(col)
    placeholders = ", ".join(f":{col}" for col in columns)
    column_list = ", ".join(columns)
    update_clause = ", ".join(
        f"{col}=EXCLUDED.{col}" for col in columns if col not in conflict_cols
    )
    conflict_list = ", ".join(conflict_cols)
    if not update_clause:
        sql = (
            f"INSERT INTO {table} ({column_list}) VALUES ({placeholders}) "
            f"ON CONFLICT ({conflict_list}) DO NOTHING"
        )
    else:
        sql = (
            f"INSERT INTO {table} ({column_list}) VALUES ({placeholders}) "
            f"ON CONFLICT ({conflict_list}) DO UPDATE SET {update_clause}"
        )
    records = frame.to_dict(orient="records")
    total = 0
    try:
        with session_scope() as session:
            for start in range(0, len(records), batch_size):
                chunk = records[start : start + batch_size]
                session.execute(text(sql), chunk)
                total += len(chunk)
    except SQLAlchemyError:
        logger.exception("upsert_failed", extra={"table": table})
        raise
    logger.info("upsert_complete", extra={"table": table, "row_count": total})
    return total


def insert_forecasts(forecast_rows: list[dict]) -> int:
    """Persist forecast rows into `demand_forecasts`."""
    if not forecast_rows:
        return 0
    frame = pd.DataFrame(forecast_rows)
    return upsert_dataframe(
        "demand_forecasts",
        frame,
        ["sku", "channel", "region", "forecast_week", "model_version"],
    )


def insert_batch_predictions(batch_rows: list[dict]) -> int:
    """Persist batch prediction rows into `batch_predictions`."""
    if not batch_rows:
        return 0
    frame = pd.DataFrame(batch_rows)
    return upsert_dataframe(
        "batch_predictions",
        frame,
        ["batch_id", "sku", "week", "channel", "region"],
    )
