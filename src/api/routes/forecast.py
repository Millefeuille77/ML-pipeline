"""Forecast endpoints."""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Annotated, Any, Final

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi import Path as FastAPIPath
from fastapi import Query, Request, status
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from config.settings import get_settings
from src.api.schemas import (
    BatchPredictionResult,
    Category,
    Channel,
    ForecastResult,
    Region,
)
from src.database.connection import get_engine, session_scope
from src.etl.pipeline import run_batch_pipeline
from src.models.forecaster import predict_horizon
from src.models.registry import ModelInfo, get_model
from src.utils.logger import get_logger
from src.utils.validators import SKU_PATTERN

logger = get_logger(__name__)

router = APIRouter(prefix="/forecast", tags=["forecast"])

_DEFAULT_HORIZON: Final[int] = 4
_MAX_HORIZON: Final[int] = 12
_LATEST_HISTORY_WEEKS: Final[int] = 26


def _load_recent_history(sku: str, channel: str, region: str) -> pd.DataFrame:
    """Load the last N weeks of history for a (sku, channel, region) triple."""
    engine = get_engine()
    sql = text(
        """
        SELECT sku, week, channel, region, units_sold, stock_available,
               promotion_flag, price_unit, delivery_days,
               is_holiday_peak, week_number, month, year,
               is_holiday_week, is_summer, is_winter, sku_age, lifecycle_stage,
               lag_1, lag_2, rolling_mean_4, rolling_std_4, momentum
        FROM weekly_features
        WHERE sku = :sku AND channel = :channel AND region = :region
        ORDER BY week DESC
        LIMIT :limit
        """
    )
    with engine.connect() as connection:
        frame = pd.read_sql(
            sql,
            connection,
            params={"sku": sku, "channel": channel, "region": region, "limit": _LATEST_HISTORY_WEEKS},
        )
    if frame.empty:
        return frame
    frame = frame.sort_values("week").reset_index(drop=True)
    return frame


def _resolve_category(sku: str) -> str:
    """Resolve the product category for a SKU; raise 404 if unknown."""
    engine = get_engine()
    with engine.connect() as connection:
        row = connection.execute(
            text("SELECT category FROM products WHERE sku = :sku"),
            {"sku": sku},
        ).fetchone()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"sku {sku!r} not found")
    return str(row[0])


def _write_prediction_log(request_id: str, forecast: ForecastResult) -> None:
    """Persist forecast points to prediction_log; failure is silent (logged only).

    Args:
        request_id: Correlation ID for the originating HTTP request.
        forecast: Completed ForecastResult to log.
    """
    rows = [
        {
            "request_id": request_id,
            "sku": forecast.sku,
            "channel": forecast.channel,
            "region": forecast.region,
            "forecast_week": week,
            "predicted_units": predicted,
            "confidence_lower": lower,
            "confidence_upper": upper,
            "model_version": forecast.model_version,
        }
        for week, predicted, lower, upper in zip(
            forecast.weeks,
            forecast.predicted_units,
            forecast.confidence_lower,
            forecast.confidence_upper,
        )
    ]
    if not rows:
        return
    sql = text(
        "INSERT INTO prediction_log "
        "(request_id, sku, channel, region, forecast_week, predicted_units, "
        "confidence_lower, confidence_upper, model_version) "
        "VALUES (:request_id, :sku, :channel, :region, :forecast_week, "
        ":predicted_units, :confidence_lower, :confidence_upper, :model_version) "
        "ON CONFLICT (request_id, sku, channel, region, forecast_week) DO NOTHING"
    )
    try:
        with session_scope() as session:
            session.execute(sql, rows)
        logger.info(
            "prediction_log_written",
            extra={"request_id": request_id, "sku": forecast.sku, "rows": len(rows)},
        )
    except SQLAlchemyError:
        logger.exception(
            "prediction_log_write_failed",
            extra={"request_id": request_id, "sku": forecast.sku},
        )


@router.get("/{sku}", response_model=ForecastResult)
def forecast_sku(
    request: Request,
    background_tasks: BackgroundTasks,
    sku: Annotated[str, FastAPIPath(pattern=SKU_PATTERN.pattern, min_length=6, max_length=6)],
    channel: Annotated[Channel, Query(description="Sales channel")],
    region: Annotated[Region, Query(description="Geographic region")],
    horizon_weeks: Annotated[int, Query(ge=1, le=_MAX_HORIZON)] = _DEFAULT_HORIZON,
) -> ForecastResult:
    """Forecast weekly demand for a single (sku, channel, region) triple."""
    category = _resolve_category(sku)
    history = _load_recent_history(sku, channel, region)
    if history.empty:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail=f"no weekly history for {sku=} {channel=} {region=}",
        )
    bundle, info = _safe_get_model(category)
    forecast = predict_horizon(bundle, sku, channel, region, history, horizon_weeks)
    forecast.model_version = f"{info.name}_v{info.version}"
    request_id = getattr(request.state, "correlation_id", None) or uuid.uuid4().hex
    background_tasks.add_task(_write_prediction_log, request_id, forecast)
    return forecast


@router.get("/category/{category}", response_model=list[ForecastResult])
def forecast_category(
    request: Request,
    background_tasks: BackgroundTasks,
    category: Annotated[Category, FastAPIPath(description="Product category")],
    channel: Annotated[Channel, Query(description="Sales channel")],
    region: Annotated[Region, Query(description="Geographic region")],
    horizon_weeks: Annotated[int, Query(ge=1, le=_MAX_HORIZON)] = _DEFAULT_HORIZON,
) -> list[ForecastResult]:
    """Forecast every SKU within a category for the given channel/region."""
    bundle, info = _safe_get_model(category)
    engine = get_engine()
    with engine.connect() as connection:
        skus = [
            row[0]
            for row in connection.execute(
                text("SELECT sku FROM products WHERE category = :category ORDER BY sku"),
                {"category": category},
            ).fetchall()
        ]
    request_id = getattr(request.state, "correlation_id", None) or uuid.uuid4().hex
    results: list[ForecastResult] = []
    for sku in skus:
        history = _load_recent_history(sku, channel, region)
        if history.empty:
            continue
        forecast = predict_horizon(bundle, sku, channel, region, history, horizon_weeks)
        forecast.model_version = f"{info.name}_v{info.version}"
        background_tasks.add_task(_write_prediction_log, request_id, forecast)
        results.append(forecast)
    return results


@router.post("/batch", response_model=BatchPredictionResult)
def forecast_batch(
    parquet_filename: Annotated[str, Query(min_length=1, max_length=200)],
) -> BatchPredictionResult:
    """Run inference on a parquet batch already staged in `RAW_DATA_DIR`.

    Args:
        parquet_filename: Filename (basename only) inside the configured raw dir.
    """
    if "/" in parquet_filename or "\\" in parquet_filename or ".." in parquet_filename:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="invalid parquet_filename")
    raw_dir = get_settings().resolved_raw_dir()
    candidate = (raw_dir / parquet_filename).resolve()
    if raw_dir.resolve() not in candidate.parents or candidate.suffix != ".parquet":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="parquet path outside raw data dir")
    if not candidate.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="parquet not found")
    return run_batch_pipeline(Path(parquet_filename))


def _safe_get_model(category: str) -> tuple[Any, ModelInfo]:
    """Load the latest forecaster bundle for `category` or raise 503."""
    name = f"forecaster_{category.lower().replace(' ', '_')}"
    try:
        return get_model(name, "latest")
    except FileNotFoundError as missing:
        logger.warning("forecast_model_missing", extra={"category": category})
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"forecast model unavailable for category {category!r}",
        ) from missing
