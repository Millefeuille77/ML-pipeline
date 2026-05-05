"""Demand forecaster: Gradient Boosting (primary) + Ridge (baseline), per category."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sqlalchemy import text

from config.settings import get_settings
from src.api.schemas import ForecastResult
from src.database.connection import session_scope
from src.models.feature_engineering import build_inference_features, build_training_features
from src.models.registry import get_model, save_model
from src.utils.logger import get_logger

logger = get_logger(__name__)

_QUANTILE_LOWER: Final[float] = 0.1
_QUANTILE_UPPER: Final[float] = 0.9
_MAPE_EPSILON: Final[float] = 1e-9
_DAYS_PER_WEEK: Final[int] = 7
_RANDOM_STATE: Final[int] = 42
# WHY: most common lifecycle in weekly_df_final_for_modeling.csv (Decline = 17811 rows)
_FALLBACK_LIFECYCLE: Final[str] = "Decline"
# WHY: ISO week numbers with PL public holidays; derived from training-data column analysis
_HOLIDAY_WEEK_NUMBERS: Final[frozenset[int]] = frozenset({14, 17, 18, 43, 44, 51, 52})
_SUMMER_MONTHS: Final[frozenset[int]] = frozenset({6, 7, 8})
_WINTER_MONTHS: Final[frozenset[int]] = frozenset({12, 1, 2})


@dataclass
class TrainedForecaster:
    """Trained per-category bundle: point + quantile estimators."""

    point: Any
    lower: Any
    upper: Any
    feature_names: list[str]
    metrics: dict[str, float] = field(default_factory=dict)
    category: str | None = None


def _gbr(loss: str, alpha: float | None = None) -> GradientBoostingRegressor:
    """Build a configured GradientBoostingRegressor from settings."""
    s = get_settings()
    kwargs: dict[str, Any] = {
        "n_estimators": s.model_n_estimators, "max_depth": s.model_max_depth,
        "learning_rate": s.model_learning_rate, "min_samples_leaf": s.model_min_samples_leaf,
        "loss": loss, "random_state": _RANDOM_STATE,
    }
    if alpha is not None:
        kwargs["alpha"] = alpha
    return GradientBoostingRegressor(**kwargs)


def _ridge() -> Ridge:
    """Build a configured Ridge regressor from settings."""
    return Ridge(alpha=get_settings().model_ridge_alpha, random_state=_RANDOM_STATE)


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return MAPE / RMSE / MAE / R² for a prediction array."""
    denom = np.where(np.abs(y_true) < _MAPE_EPSILON, _MAPE_EPSILON, np.abs(y_true))
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else 0.0
    return {"mape": mape, "rmse": rmse, "mae": mae, "r2": r2}


def train_gbr(features: pd.DataFrame, target: pd.Series) -> TrainedForecaster:
    """Train a quantile-regression GBR triple (lower / point / upper)."""
    point = _gbr(loss="squared_error")
    lower = _gbr(loss="quantile", alpha=_QUANTILE_LOWER)
    upper = _gbr(loss="quantile", alpha=_QUANTILE_UPPER)
    point.fit(features, target)
    lower.fit(features, target)
    upper.fit(features, target)
    metrics = _evaluate(target.to_numpy(), point.predict(features))
    return TrainedForecaster(
        point=point, lower=lower, upper=upper,
        feature_names=list(features.columns), metrics=metrics,
    )


def train_ridge(features: pd.DataFrame, target: pd.Series) -> TrainedForecaster:
    """Train the Ridge baseline; widen intervals using residual std."""
    estimator = _ridge()
    estimator.fit(features, target)
    predictions = estimator.predict(features)
    residual_std = float(np.std(target.to_numpy() - predictions))
    interval = 1.2816 * residual_std  # one-sided z-value at the 0.1 / 0.9 quantiles
    metrics = _evaluate(target.to_numpy(), predictions)
    return TrainedForecaster(
        point=estimator, lower=("residual_offset", -interval),
        upper=("residual_offset", interval),
        feature_names=list(features.columns), metrics=metrics,
    )


def train_per_category(weekly_df: pd.DataFrame) -> dict[str, TrainedForecaster]:
    """Train one GBR forecaster per category, persisting each via the registry.

    Raises:
        ValueError: If the input lacks a `category` column.
    """
    if "category" not in weekly_df.columns:
        raise ValueError("weekly_df missing required 'category' column.")
    forecasters: dict[str, TrainedForecaster] = {}
    for category, subset in weekly_df.groupby("category"):
        features, target = build_training_features(subset)
        if features.empty:
            logger.warning("forecaster_skipped_empty_category", extra={"category": category})
            continue
        bundle = train_gbr(features, target)
        bundle.category = str(category)
        save_model(
            bundle, name=f"forecaster_{_normalize(category)}",
            metrics=bundle.metrics, features=bundle.feature_names,
            training_rows=len(features), category=str(category),
        )
        forecasters[str(category)] = bundle
    return forecasters


def _normalize(category: str) -> str:
    """Lower-case category label for use in registry filenames."""
    return str(category).lower().replace(" ", "_")


def _predict_with_intervals(
    bundle: TrainedForecaster, features: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run point + interval prediction aligned to bundle feature names."""
    aligned = features.reindex(columns=bundle.feature_names, fill_value=0.0)
    point = bundle.point.predict(aligned)
    if isinstance(bundle.lower, tuple) and bundle.lower[0] == "residual_offset":
        lower = point + float(bundle.lower[1])
        upper = point + float(bundle.upper[1])
    else:
        lower = bundle.lower.predict(aligned)
        upper = bundle.upper.predict(aligned)
    return point, np.minimum(lower, point), np.maximum(upper, point)


def _grow_history_one_step(
    history: pd.DataFrame, target_week: date, predicted: float
) -> pd.DataFrame:
    """Append a synthetic row so the next step has updated lag/rolling features."""
    next_row = history.iloc[-1].copy()
    next_row["week"] = target_week
    next_row["lag_2"] = next_row.get("lag_1", predicted)
    next_row["lag_1"] = predicted
    next_row["units_sold"] = predicted
    window = pd.concat([history.tail(3)["units_sold"], pd.Series([predicted])], ignore_index=True)
    next_row["rolling_mean_4"] = float(window.mean())
    next_row["rolling_std_4"] = float(window.std() or 0.0)
    next_row["momentum"] = float(next_row["lag_1"] - next_row.get("lag_2", predicted))
    return pd.concat([history, pd.DataFrame([next_row])], ignore_index=True)


def predict_horizon(
    bundle: TrainedForecaster, sku: str, channel: str, region: str,
    recent_data: pd.DataFrame, horizon_weeks: int,
) -> ForecastResult:
    """Roll-forward predict `horizon_weeks` weeks, updating lags at each step."""
    history = recent_data.copy()
    weeks_ahead: list[date] = []
    point_units: list[float] = []
    lower_units: list[float] = []
    upper_units: list[float] = []
    last_week = pd.to_datetime(history["week"].max()).date()
    for step in range(1, horizon_weeks + 1):
        feature_row = build_inference_features(sku, channel, region, history)
        point, lower, upper = _predict_with_intervals(bundle, feature_row)
        target_week = last_week + timedelta(days=_DAYS_PER_WEEK * step)
        weeks_ahead.append(target_week)
        point_units.append(float(point[0]))
        lower_units.append(float(lower[0]))
        upper_units.append(float(upper[0]))
        history = _grow_history_one_step(history, target_week, float(point[0]))
    return ForecastResult(
        sku=sku, channel=channel,  # type: ignore[arg-type]
        region=region,  # type: ignore[arg-type]
        weeks=weeks_ahead, predicted_units=point_units,
        confidence_lower=lower_units, confidence_upper=upper_units,
        model_version=f"gbr_{bundle.category or 'mixed'}",
    )


def predict_batch_dataframe(batch_df: pd.DataFrame) -> list[ForecastResult]:
    """Score a daily-granularity batch parquet; return one ForecastResult per group."""
    results: list[ForecastResult] = []
    if batch_df.empty:
        return results
    horizon = get_settings().model_forecast_horizon_weeks
    weekly_history = _aggregate_to_weekly_history(batch_df)
    for (sku, channel, region), group in weekly_history.groupby(["sku", "channel", "region"]):
        category = str(group["category"].iloc[0]) if "category" in group.columns else "Milk"
        try:
            bundle, _info = get_model(f"forecaster_{_normalize(category)}", "latest")
        except FileNotFoundError:
            logger.warning("batch_predict_no_model", extra={"category": category, "sku": sku})
            continue
        recent = group.sort_values("week").reset_index(drop=True)
        results.append(predict_horizon(bundle, str(sku), str(channel), str(region), recent, horizon))
    return results


def _resample_weekly(daily_frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily rows to weekly grain with week/month/year columns."""
    frame = daily_frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["week"] = frame["date"].dt.to_period("W-SUN").dt.start_time.dt.date
    extra = [c for c in ("category", "brand", "segment", "pack_type") if c in frame.columns]
    agg = frame.groupby(["sku", "channel", "region", "week", *extra], as_index=False).agg(
        units_sold=("units_sold", "sum"), stock_available=("stock_available", "max"),
        promotion_flag=("promotion_flag", "max"), price_unit=("price_unit", "mean"),
        delivery_days=("delivery_days", "mean"),
    )
    wk = pd.to_datetime(agg["week"])
    agg["week_number"] = wk.dt.isocalendar().week.astype(int)
    agg["month"] = wk.dt.month.astype(int)
    agg["year"] = wk.dt.year.astype(int)
    return agg


def _lookup_sku_metadata(sku: str) -> tuple[str, int]:
    """Query DB for latest lifecycle_stage and sku_age for a SKU.

    WHY: lifecycle_stage is an SKU-level attribute that changes over time; reading
    from weekly_features at prediction time avoids propagating stale training-time
    labels into inference, which would systematically bias Growth/Decline SKUs.
    """
    try:
        with session_scope() as session:
            row = session.execute(
                text("SELECT lifecycle_stage, sku_age FROM weekly_features "
                     "WHERE sku = :sku ORDER BY week DESC LIMIT 1"),
                {"sku": sku},
            ).fetchone()
        if row is not None:
            return str(row[0]), int(row[1])
    except Exception:
        logger.warning("lifecycle_db_lookup_failed", extra={"sku": sku})
    logger.warning("lifecycle_fallback_used", extra={"sku": sku, "fallback": _FALLBACK_LIFECYCLE})
    return _FALLBACK_LIFECYCLE, 1


def _stub_modeling_columns(weekly_frame: pd.DataFrame, sku: str) -> pd.DataFrame:
    """Populate lag/rolling/calendar columns on a weekly-aggregated batch frame."""
    lifecycle, sku_age_base = _lookup_sku_metadata(sku)
    frame = weekly_frame.copy()
    frame["lag_1"] = 0.0
    frame["lag_2"] = 0.0
    frame["rolling_mean_4"] = frame["units_sold"]
    frame["rolling_std_4"] = 0.0
    frame["momentum"] = 0.0
    wn = frame["week_number"]
    frame["is_holiday_week"] = wn.isin(_HOLIDAY_WEEK_NUMBERS).astype(int)
    frame["is_holiday_peak"] = frame["is_holiday_week"]
    frame["is_summer"] = frame["month"].isin(_SUMMER_MONTHS).astype(int)
    frame["is_winter"] = frame["month"].isin(_WINTER_MONTHS).astype(int)
    frame["sku_age"] = sku_age_base + frame.index
    frame["lifecycle_stage"] = lifecycle
    return frame


def _aggregate_to_weekly_history(batch_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a daily batch to weekly granularity with correct modeling columns."""
    aggregated = _resample_weekly(batch_df)
    parts = [_stub_modeling_columns(aggregated[aggregated["sku"] == s], s)
             for s in aggregated["sku"].unique()]
    return pd.concat(parts, ignore_index=True) if parts else aggregated


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Public re-export of the metric helper for evaluation modules."""
    return _evaluate(y_true, y_pred)
