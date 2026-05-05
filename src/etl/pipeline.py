"""High-level pipeline runners (full ETL and per-batch inference)."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import pandas as pd

from config.settings import get_settings
from src.api.schemas import BatchPredictionResult, ForecastResult
from src.etl.extractors import (
    extract_batch_parquet,
    extract_daily_csv,
    extract_enriched_csv,
    extract_weekly_csv,
)
from src.etl.loaders import insert_batch_predictions, upsert_dataframe
from src.etl.transformers import (
    aggregate_to_weekly,
    clean_daily_data,
    enrich_features,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

_DAILY_FILE = "FMCG_2022_2024.csv"
_WEEKLY_FILE = "weekly_df_final_for_modeling.csv"
_ENRICHED_FILE = "df_weekly_MI-006_enriched.csv"


def run_full_pipeline() -> dict[str, int]:
    """Run the full CSV → clean → enrich → load pipeline.

    Returns:
        Dict of stage names → row counts processed.
    """
    settings = get_settings()
    raw_dir = settings.resolved_raw_dir()
    started = perf_counter()
    daily_df = extract_daily_csv(raw_dir / _DAILY_FILE)
    weekly_df = extract_weekly_csv(raw_dir / _WEEKLY_FILE)
    enriched_df = extract_enriched_csv(raw_dir / _ENRICHED_FILE)
    cleaned_daily = clean_daily_data(daily_df)
    weekly_agg = aggregate_to_weekly(cleaned_daily)
    enriched_all = enrich_features(weekly_agg, enriched_df)
    counts = {
        "daily_rows": len(cleaned_daily),
        "weekly_rows_aggregated": len(weekly_agg),
        "weekly_rows_modeling": len(weekly_df),
        "enriched_rows": len(enriched_all),
    }
    _write_processed_artifacts(enriched_all, weekly_df, settings.resolved_processed_dir())
    elapsed = perf_counter() - started
    logger.info(
        "full_pipeline_complete",
        extra={"elapsed_seconds": round(elapsed, 2), **counts},
    )
    return counts


def _write_processed_artifacts(
    enriched: pd.DataFrame, weekly: pd.DataFrame, processed_dir: Path
) -> None:
    """Write parquet snapshots of derived weekly artifacts."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(processed_dir / "weekly_enriched.parquet", index=False)
    weekly.to_parquet(processed_dir / "weekly_modeling.parquet", index=False)


def run_batch_pipeline(parquet_path: Path) -> BatchPredictionResult:
    """Read a weekly batch parquet, run inference, persist predictions.

    Args:
        parquet_path: Path to the batch parquet file (must be inside RAW_DATA_DIR).

    Returns:
        BatchPredictionResult envelope containing per-SKU forecasts.

    Raises:
        ValueError: If `parquet_path` escapes the configured raw data dir.
        FileNotFoundError: If the file does not exist.
    """
    settings = get_settings()
    safe_path = _resolve_within(parquet_path, settings.resolved_raw_dir())
    frame = extract_batch_parquet(safe_path)
    from src.models.forecaster import predict_batch_dataframe  # local import: break cycle
    started = perf_counter()
    predictions: list[ForecastResult] = predict_batch_dataframe(frame)
    batch_id = f"batch-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:6]}"
    rows = _flatten_batch_predictions(batch_id, predictions)
    insert_batch_predictions(rows)
    elapsed = perf_counter() - started
    logger.info(
        "batch_pipeline_complete",
        extra={
            "batch_id": batch_id,
            "elapsed_seconds": round(elapsed, 2),
            "prediction_count": len(predictions),
        },
    )
    return BatchPredictionResult(
        batch_id=batch_id,
        predictions=predictions,
        created_at=datetime.now(timezone.utc),
    )


def _resolve_within(candidate: Path, root: Path) -> Path:
    """Return `candidate` resolved if it sits inside `root`, else raise."""
    resolved = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    root_resolved = root.resolve()
    if root_resolved not in resolved.parents and resolved != root_resolved:
        raise ValueError(f"path escapes raw data dir: {candidate}")
    if not resolved.exists():
        raise FileNotFoundError(f"batch file not found: {candidate}")
    return resolved


def _flatten_batch_predictions(
    batch_id: str, predictions: list[ForecastResult]
) -> list[dict]:
    """Flatten ForecastResult objects into per-week DB rows."""
    rows: list[dict] = []
    for forecast in predictions:
        for index, week in enumerate(forecast.weeks):
            rows.append(
                {
                    "batch_id": batch_id,
                    "sku": forecast.sku,
                    "week": week,
                    "channel": forecast.channel,
                    "region": forecast.region,
                    "predicted_units": forecast.predicted_units[index],
                    "confidence_lower": forecast.confidence_lower[index],
                    "confidence_upper": forecast.confidence_upper[index],
                    "model_version": forecast.model_version,
                }
            )
    return rows


def persist_processed_weekly(weekly_df: pd.DataFrame) -> int:
    """Upsert processed weekly modeling rows into `weekly_features`."""
    return upsert_dataframe(
        "weekly_features", weekly_df, ["sku", "week", "channel", "region"]
    )
