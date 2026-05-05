"""Extractors: pure read functions returning DataFrames.

No transformations, no I/O side-effects beyond reading the input source.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import text

from src.database.connection import get_engine
from src.utils.logger import get_logger

logger = get_logger(__name__)


def extract_daily_csv(path: Path) -> pd.DataFrame:
    """Read the daily fact CSV.

    Args:
        path: Filesystem path to the daily CSV.

    Returns:
        DataFrame with columns matching `daily_sales` (plus product master cols).

    Raises:
        FileNotFoundError: If `path` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Daily CSV not found: {path}")
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    logger.info("daily_csv_loaded", extra={"row_count": len(frame), "path": str(path)})
    return frame


def extract_weekly_csv(path: Path) -> pd.DataFrame:
    """Read the weekly modeling CSV.

    Args:
        path: Filesystem path to the weekly CSV.

    Returns:
        DataFrame with weekly modeling columns including pre-computed lag features.

    Raises:
        FileNotFoundError: If `path` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Weekly CSV not found: {path}")
    frame = pd.read_csv(path)
    frame["week"] = pd.to_datetime(frame["week"]).dt.date
    logger.info("weekly_csv_loaded", extra={"row_count": len(frame), "path": str(path)})
    return frame


def extract_enriched_csv(path: Path) -> pd.DataFrame:
    """Read the MI-006 enrichment CSV (template for all-SKU generalization)."""
    if not path.exists():
        raise FileNotFoundError(f"Enriched CSV not found: {path}")
    frame = pd.read_csv(path)
    frame["week"] = pd.to_datetime(frame["week"]).dt.date
    logger.info("enriched_csv_loaded", extra={"row_count": len(frame), "path": str(path)})
    return frame


def extract_batch_parquet(path: Path) -> pd.DataFrame:
    """Read a weekly batch parquet file.

    Args:
        path: Filesystem path to the parquet batch file.

    Returns:
        DataFrame matching the daily fact schema.

    Raises:
        FileNotFoundError: If `path` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Batch parquet not found: {path}")
    frame = pd.read_parquet(path)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"]).dt.date
    logger.info("batch_parquet_loaded", extra={"row_count": len(frame), "path": str(path)})
    return frame


def extract_from_db(query: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Execute a parameterized SQL query and return the result as a DataFrame.

    Args:
        query: SQL string with bind params (`:name`-style placeholders).
        params: Mapping of bind parameter names to values.

    Returns:
        DataFrame containing the query results.
    """
    engine = get_engine()
    with engine.connect() as connection:
        frame = pd.read_sql(text(query), connection, params=params or {})
    logger.info("db_query_extracted", extra={"row_count": len(frame)})
    return frame
