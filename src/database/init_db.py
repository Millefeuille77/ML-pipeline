"""Initialize the FMCG database: create schema and load all raw datasets.

Run with: ``python -m src.database.init_db``
Idempotent — safe to re-execute (DROP+CREATE for schema, ON CONFLICT for rows).
"""
from __future__ import annotations

import re
from pathlib import Path
from time import perf_counter
from typing import Iterable

import pandas as pd
from sqlalchemy import text

from config.settings import get_settings
from src.database.connection import get_engine, shutdown_engine
from src.etl.loaders import upsert_dataframe
from src.utils.logger import get_logger

logger = get_logger(__name__)

_DAILY_FILE = "FMCG_2022_2024.csv"
_WEEKLY_FILE = "weekly_df_final_for_modeling.csv"
_ENRICHED_FILE = "df_weekly_MI-006_enriched.csv"
_BATCH_INSERT_SIZE = 5000


def _read_schema_sql() -> str:
    """Read the schema.sql file shipped beside this module."""
    schema_path = Path(__file__).resolve().parent / "schema.sql"
    return schema_path.read_text(encoding="utf-8")


def create_schema() -> None:
    """Execute schema.sql to (re)create all tables."""
    sql = _read_schema_sql()
    engine = get_engine()
    with engine.begin() as connection:
        for statement in _split_sql_statements(sql):
            connection.execute(text(statement))
    logger.info("schema_created")


def _split_sql_statements(script: str) -> Iterable[str]:
    """Split a SQL script on `;` while ignoring blank/comment-only chunks.

    WHY: a previous implementation only skipped chunks whose FIRST stripped
    line started with `--`. Statements preceded by header comments were
    silently dropped, breaking idempotency on re-runs.
    """
    no_line_comments = re.sub(r"--[^\n]*", "", script)
    for raw in no_line_comments.split(";"):
        stripped = raw.strip()
        if stripped:
            yield stripped


def _bulk_insert(table: str, frame: pd.DataFrame, conflict_cols: list[str]) -> int:
    """Idempotent bulk insert delegating to `loaders.upsert_dataframe`.

    Args:
        table: Target table name (must be in loaders allow-list).
        frame: DataFrame whose columns map 1:1 to table columns.
        conflict_cols: Columns forming the unique conflict target.

    Returns:
        Number of rows inserted/upserted.
    """
    return upsert_dataframe(table, frame, conflict_cols, batch_size=_BATCH_INSERT_SIZE)


def load_products(daily_df: pd.DataFrame) -> int:
    """Derive distinct product master rows from the daily fact CSV."""
    products = daily_df[["sku", "brand", "segment", "category", "pack_type"]].drop_duplicates(
        subset=["sku"]
    ).reset_index(drop=True)
    inserted = _bulk_insert("products", products, ["sku"])
    logger.info("products_loaded", extra={"row_count": inserted})
    return inserted


def load_daily_sales(daily_df: pd.DataFrame) -> int:
    """Load the daily fact table from `FMCG_2022_2024.csv`."""
    frame = daily_df.rename(columns={"date": "sale_date"}).copy()
    keep_cols = [
        "sku", "sale_date", "channel", "region", "price_unit", "promotion_flag",
        "delivery_days", "stock_available", "delivered_qty", "units_sold",
    ]
    frame = frame[keep_cols]
    frame["sale_date"] = pd.to_datetime(frame["sale_date"]).dt.date
    inserted = _bulk_insert(
        "daily_sales", frame, ["sku", "sale_date", "channel", "region"]
    )
    logger.info("daily_sales_loaded", extra={"row_count": inserted})
    return inserted


def load_weekly_features(weekly_df: pd.DataFrame) -> int:
    """Load the weekly modeling table from `weekly_df_final_for_modeling.csv`."""
    frame = weekly_df.copy()
    frame["week"] = pd.to_datetime(frame["week"]).dt.date
    # WHY: pandas infers `is_holiday_peak` as bool from the CSV; schema declares
    # SMALLINT for consistency with the other holiday/season flags.
    if frame["is_holiday_peak"].dtype == bool:
        frame["is_holiday_peak"] = frame["is_holiday_peak"].astype("int8")
    inserted = _bulk_insert("weekly_features", frame, ["sku", "week", "channel", "region"])
    logger.info("weekly_features_loaded", extra={"row_count": inserted})
    return inserted


def load_enrichment_features(enriched_df: pd.DataFrame) -> int:
    """Load enrichment columns from `df_weekly_MI-006_enriched.csv`."""
    keep = [
        "sku", "week", "channel", "region", "price_avg", "promo_rate", "stock_avg",
        "deliveries", "avg_temp", "inflation_index", "school_in_session",
        "category_trend", "event_score",
    ]
    frame = enriched_df[keep].copy()
    frame["week"] = pd.to_datetime(frame["week"]).dt.date
    inserted = _bulk_insert(
        "enrichment_features", frame, ["sku", "week", "channel", "region"]
    )
    logger.info("enrichment_features_loaded", extra={"row_count": inserted})
    return inserted


def main() -> None:
    """Entry point: create schema and load all raw datasets."""
    settings = get_settings()
    raw_dir = settings.resolved_raw_dir()
    started = perf_counter()
    create_schema()
    daily_df = pd.read_csv(raw_dir / _DAILY_FILE)
    weekly_df = pd.read_csv(raw_dir / _WEEKLY_FILE)
    enriched_df = pd.read_csv(raw_dir / _ENRICHED_FILE)
    load_products(daily_df)
    load_daily_sales(daily_df)
    load_weekly_features(weekly_df)
    load_enrichment_features(enriched_df)
    elapsed = perf_counter() - started
    logger.info("init_db_complete", extra={"elapsed_seconds": round(elapsed, 2)})
    shutdown_engine()


if __name__ == "__main__":  # pragma: no cover
    from config.logging_config import configure_logging
    configure_logging(get_settings().app_log_level)
    main()
