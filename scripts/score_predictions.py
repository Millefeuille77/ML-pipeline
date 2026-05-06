"""Daily scoring job: backfill actual_units in prediction_log and compute live metrics.

Usage:
    python -m scripts.score_predictions [--dry-run]

What it does:
1. Find prediction_log rows where forecast_week <= TODAY - 7 days AND scored_at IS NULL.
2. Join with daily_sales aggregated to (sku, channel, region, week) to obtain actual units.
3. Update prediction_log.actual_units and scored_at = NOW() for matched rows.
4. Aggregate per (model_version, category, scored_date=TODAY) to compute live MAPE/RMSE/MAE.
5. UPSERT into model_performance_live.
Idempotent: re-running the same day overwrites the same (scored_date, model_version, category).
"""
from __future__ import annotations

import argparse
import math
import sys
from datetime import date, timedelta
from pathlib import Path

# Allow `python -m scripts.score_predictions` from project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from src.database.connection import get_engine, session_scope
from src.utils.logger import get_logger

logger = get_logger(__name__)

_SETTLE_DAYS: int = 7


def _fetch_unscored(engine) -> pd.DataFrame:
    """Return prediction_log rows eligible for scoring.

    Args:
        engine: SQLAlchemy engine.

    Returns:
        DataFrame of unscored prediction rows joined with product category.
    """
    cutoff = date.today() - timedelta(days=_SETTLE_DAYS)
    sql = text(
        "SELECT pl.id, pl.sku, pl.channel, pl.region, pl.forecast_week, "
        "       pl.predicted_units, pl.model_version, p.category "
        "FROM prediction_log pl "
        "JOIN products p ON p.sku = pl.sku "
        "WHERE pl.forecast_week <= :cutoff AND pl.scored_at IS NULL"
    )
    with engine.connect() as connection:
        return pd.read_sql(sql, connection, params={"cutoff": cutoff})


def _fetch_actuals(engine, skus: list[str], forecast_weeks: list[date]) -> pd.DataFrame:
    """Aggregate daily_sales to weekly actuals for the relevant (sku, channel, region, week).

    Args:
        engine: SQLAlchemy engine.
        skus: Distinct SKU list to filter on.
        forecast_weeks: Distinct forecast weeks (Monday-anchored dates) to filter on.

    Returns:
        DataFrame with columns sku, channel, region, week_start, actual_units.
    """
    if not skus or not forecast_weeks:
        return pd.DataFrame(columns=["sku", "channel", "region", "week_start", "actual_units"])
    sql = text(
        "SELECT sku, channel, region, "
        "       DATE_TRUNC('week', sale_date)::DATE AS week_start, "
        "       SUM(units_sold) AS actual_units "
        "FROM daily_sales "
        "WHERE sku = ANY(:skus) "
        "  AND DATE_TRUNC('week', sale_date)::DATE = ANY(:weeks) "
        "GROUP BY sku, channel, region, week_start"
    )
    with engine.connect() as connection:
        return pd.read_sql(
            sql,
            connection,
            params={"skus": skus, "weeks": forecast_weeks},
        )


def _compute_metrics(joined: pd.DataFrame) -> pd.DataFrame:
    """Compute MAPE, RMSE, MAE per (model_version, category).

    Args:
        joined: Rows with predicted_units, actual_units, model_version, category.

    Returns:
        DataFrame with columns model_version, category, samples, mape, rmse, mae.
    """
    records = []
    for (model_version, category), group in joined.groupby(["model_version", "category"]):
        predicted = group["predicted_units"].values
        actual = group["actual_units"].values
        samples = len(group)
        mae = float(abs(actual - predicted).mean())
        rmse = float(math.sqrt(((actual - predicted) ** 2).mean()))
        non_zero_mask = actual != 0
        if non_zero_mask.sum() > 0:
            mape = float((abs((actual[non_zero_mask] - predicted[non_zero_mask]) / actual[non_zero_mask])).mean())
        else:
            mape = 0.0
        records.append({
            "model_version": model_version,
            "category": category,
            "samples": samples,
            "live_mape": mape,
            "live_rmse": rmse,
            "live_mae": mae,
        })
    return pd.DataFrame(records)


def _bulk_update_rows(rows: list[dict], dry_run: bool) -> None:
    """Update prediction_log rows with actual_units and scored_at.

    Args:
        rows: List of dicts with keys id, actual_units.
        dry_run: If True, skip writes and log only.
    """
    if not rows:
        return
    if dry_run:
        logger.info("dry_run_skip_update", extra={"rows": len(rows)})
        return
    sql = text(
        "UPDATE prediction_log SET actual_units = :actual_units, scored_at = NOW() WHERE id = :id"
    )
    try:
        with session_scope() as session:
            session.execute(sql, rows)
        logger.info("prediction_log_scored", extra={"rows": len(rows)})
    except SQLAlchemyError:
        logger.exception("prediction_log_update_failed")
        raise


def _upsert_performance(metrics: pd.DataFrame, scored_date: date, dry_run: bool) -> None:
    """UPSERT aggregated metrics into model_performance_live.

    Args:
        metrics: DataFrame from _compute_metrics.
        scored_date: The date to record (today).
        dry_run: If True, skip writes.
    """
    if metrics.empty:
        return
    if dry_run:
        logger.info("dry_run_skip_perf_upsert", extra={"rows": len(metrics)})
        return
    sql = text(
        "INSERT INTO model_performance_live "
        "(scored_date, model_version, category, samples, live_mape, live_rmse, live_mae) "
        "VALUES (:scored_date, :model_version, :category, :samples, :live_mape, :live_rmse, :live_mae) "
        "ON CONFLICT (scored_date, model_version, category) DO UPDATE SET "
        "samples=EXCLUDED.samples, live_mape=EXCLUDED.live_mape, "
        "live_rmse=EXCLUDED.live_rmse, live_mae=EXCLUDED.live_mae"
    )
    records = metrics.assign(scored_date=scored_date).to_dict(orient="records")
    try:
        with session_scope() as session:
            session.execute(sql, records)
        logger.info("performance_upserted", extra={"rows": len(records), "scored_date": str(scored_date)})
    except SQLAlchemyError:
        logger.exception("performance_upsert_failed")
        raise


def run_scoring(dry_run: bool = False) -> None:
    """Full scoring pass: fetch unscored predictions, join actuals, update and persist metrics.

    Args:
        dry_run: If True, compute metrics but skip all writes.
    """
    engine = get_engine()
    logger.info("scoring_start", extra={"dry_run": dry_run})

    unscored = _fetch_unscored(engine)
    if unscored.empty:
        logger.info("scoring_no_unscored_rows")
        return
    logger.info("scoring_unscored_rows", extra={"count": len(unscored)})

    distinct_skus = unscored["sku"].unique().tolist()
    distinct_weeks = [d.date() if hasattr(d, "date") else d for d in unscored["forecast_week"].unique().tolist()]
    actuals = _fetch_actuals(engine, distinct_skus, distinct_weeks)

    merge_keys = ["sku", "channel", "region"]
    actuals = actuals.rename(columns={"week_start": "forecast_week"})
    joined = unscored.merge(actuals, on=merge_keys + ["forecast_week"], how="inner")
    logger.info("scoring_matched_rows", extra={"matched": len(joined), "total_unscored": len(unscored)})

    if joined.empty:
        logger.info("scoring_no_actuals_matched")
        return

    update_rows = [
        {"id": int(row["id"]), "actual_units": float(row["actual_units"])}
        for _, row in joined.iterrows()
    ]
    _bulk_update_rows(update_rows, dry_run)

    metrics = _compute_metrics(joined)
    _upsert_performance(metrics, date.today(), dry_run)
    logger.info("scoring_complete", extra={"metrics_rows": len(metrics)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score prediction_log against actuals.")
    parser.add_argument("--dry-run", action="store_true", help="Compute but skip writes.")
    args = parser.parse_args()
    run_scoring(dry_run=args.dry_run)
