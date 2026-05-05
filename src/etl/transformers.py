"""Transformers: pure DataFrame in/out, no I/O.

Negative `units_sold` and `delivered_qty` are valid (returns) — never filtered.
"""
from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from src.utils.logger import get_logger
from src.utils.validators import (
    VALID_CATEGORIES,
    VALID_CHANNELS,
    VALID_LIFECYCLE_STAGES,
    VALID_REGIONS,
)

logger = get_logger(__name__)

_REQUIRED_DAILY_COLS: Final[tuple[str, ...]] = (
    "date", "sku", "brand", "segment", "category", "channel", "region",
    "pack_type", "price_unit", "promotion_flag", "delivery_days",
    "stock_available", "delivered_qty", "units_sold",
)

_DELIVERY_DAYS_MIN: Final[int] = 1
_DELIVERY_DAYS_MAX: Final[int] = 5
_PRICE_UNIT_MIN: Final[float] = 1e-6


def _log_enum_violations(frame: pd.DataFrame) -> None:
    """Warn when category, channel, or region values are outside the known sets."""
    invalid = (
        ~frame["category"].isin(VALID_CATEGORIES)
        | ~frame["channel"].isin(VALID_CHANNELS)
        | ~frame["region"].isin(VALID_REGIONS)
    )
    if invalid.any():
        logger.warning("daily_clean_unknown_enums", extra={"bad_row_count": int(invalid.sum())})


def _log_price_violations(frame: pd.DataFrame) -> None:
    """Warn when price_unit is at or below the minimum threshold."""
    bad_mask = frame["price_unit"].le(_PRICE_UNIT_MIN)
    if bad_mask.any():
        logger.warning(
            "daily_clean_non_positive_price", extra={"bad_row_count": int(bad_mask.sum())}
        )


def _log_delivery_range(frame: pd.DataFrame) -> None:
    """Warn when delivery_days is outside the allowed [1, 5] range."""
    out_of_range = (
        (frame["delivery_days"] < _DELIVERY_DAYS_MIN)
        | (frame["delivery_days"] > _DELIVERY_DAYS_MAX)
    )
    if out_of_range.any():
        logger.warning(
            "daily_clean_delivery_out_of_range",
            extra={"bad_row_count": int(out_of_range.sum())},
        )


def clean_daily_data(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Validate, type-coerce, and flag (don't drop) anomalies.

    Args:
        daily_df: Raw daily DataFrame from `extract_daily_csv`.

    Returns:
        Cleaned DataFrame with `is_return` boolean flag added.

    Raises:
        ValueError: If a required column is missing.
    """
    missing = set(_REQUIRED_DAILY_COLS) - set(daily_df.columns)
    if missing:
        raise ValueError(f"daily DataFrame missing columns: {sorted(missing)}")
    frame = daily_df.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date
    _log_enum_violations(frame)
    frame["is_return"] = frame["units_sold"] < 0
    _log_price_violations(frame)
    _log_delivery_range(frame)
    return frame


def aggregate_to_weekly(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the daily fact table to weekly granularity.

    Args:
        daily_df: Cleaned daily DataFrame.

    Returns:
        Weekly DataFrame keyed by (sku, week, channel, region).
    """
    frame = daily_df.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["week"] = frame["date"].dt.to_period("W-SUN").dt.start_time.dt.date
    grouped = frame.groupby(["sku", "week", "channel", "region"], as_index=False).agg(
        units_sold=("units_sold", "sum"),
        delivered_qty=("delivered_qty", "sum"),
        stock_available=("stock_available", "max"),
        promotion_flag=("promotion_flag", "max"),
        price_unit=("price_unit", "mean"),
        delivery_days=("delivery_days", "mean"),
    )
    return grouped


def _template_weekly(template_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate MI-006 template to one row per week with mean/max of enrichment cols."""
    return (
        template_df.groupby("week", as_index=False)
        .agg(
            avg_temp=("avg_temp", "mean"),
            inflation_index=("inflation_index", "mean"),
            school_in_session=("school_in_session", "max"),
            category_trend=("category_trend", "mean"),
            event_score=("event_score", "mean"),
        )
    )


def _apply_enrichment_defaults(
    frame: pd.DataFrame, template_df: pd.DataFrame
) -> pd.DataFrame:
    """Left-join weekly template and fill NaNs with column-level means/modes."""
    fill_defaults = {
        "avg_temp": template_df["avg_temp"].mean(),
        "inflation_index": template_df["inflation_index"].mean(),
        "school_in_session": int(template_df["school_in_session"].mode().iloc[0]),
        "category_trend": template_df["category_trend"].mean(),
        "event_score": template_df["event_score"].mean(),
    }
    for column, default in fill_defaults.items():
        frame[column] = frame[column].fillna(default)
    return frame


def enrich_features(weekly_df: pd.DataFrame, template_df: pd.DataFrame) -> pd.DataFrame:
    """Generalize the 9 enrichment columns (MI-006 template) to all SKUs.

    Args:
        weekly_df: Weekly aggregate DataFrame.
        template_df: MI-006 enriched DataFrame used as a reference distribution.

    Returns:
        Weekly DataFrame plus enrichment columns.
    """
    frame = weekly_df.copy()
    frame["price_avg"] = frame.groupby(["sku", "channel", "region"])["price_unit"].transform("mean")
    frame["promo_rate"] = frame["promotion_flag"].astype(float).clip(0.0, 1.0)
    frame["stock_avg"] = frame.groupby(["sku", "channel", "region"])["stock_available"].transform("mean")
    if "delivered_qty" in frame.columns:
        frame["deliveries"] = (frame["delivered_qty"] > 0).astype(int)
    else:
        frame["deliveries"] = 0
    weekly_template = _template_weekly(template_df)
    frame["week"] = pd.to_datetime(frame["week"]).dt.date
    weekly_template["week"] = pd.to_datetime(weekly_template["week"]).dt.date
    enriched = frame.merge(weekly_template, on="week", how="left")
    return _apply_enrichment_defaults(enriched, template_df)


_LAG_COLS: Final[tuple[str, ...]] = (
    "lag_1", "lag_2", "rolling_mean_4", "rolling_std_4", "momentum"
)


def _compute_lags(units: pd.Series) -> pd.DataFrame:
    """Return lag_1 and lag_2 for a units_sold Series (within one group)."""
    return pd.DataFrame({"lag_1": units.shift(1), "lag_2": units.shift(2)})


def _compute_rolling(units: pd.Series) -> pd.DataFrame:
    """Return rolling mean/std for a units_sold Series (within one group)."""
    shifted = units.shift(1)
    return pd.DataFrame({
        "rolling_mean_4": shifted.rolling(window=4, min_periods=1).mean(),
        "rolling_std_4": shifted.rolling(window=4, min_periods=1).std(),
    })


def _fill_and_validate_lags(frame: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill series-start NaNs, zero-fill any remainder, warn on lifecycle."""
    frame[list(_LAG_COLS)] = frame.groupby(["sku", "channel", "region"])[list(_LAG_COLS)].transform(
        lambda series: series.ffill().bfill()
    )
    frame[list(_LAG_COLS)] = frame[list(_LAG_COLS)].fillna(0.0)
    if "lifecycle_stage" in frame.columns:
        invalid = ~frame["lifecycle_stage"].isin(VALID_LIFECYCLE_STAGES)
        if invalid.any():
            logger.warning("lag_invalid_lifecycle_rows", extra={"bad_row_count": int(invalid.sum())})
    frame.replace({np.inf: 0.0, -np.inf: 0.0}, inplace=True)
    return frame


def compute_lag_features(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """Compute lag, rolling, and momentum features per (sku, channel, region).

    Args:
        weekly_df: Weekly aggregate DataFrame sorted (or sortable) by week.

    Returns:
        DataFrame with `lag_1`, `lag_2`, `rolling_mean_4`, `rolling_std_4`,
        `momentum` columns.
    """
    frame = weekly_df.sort_values(["sku", "channel", "region", "week"]).copy()
    grouped = frame.groupby(["sku", "channel", "region"], group_keys=False)
    lags = grouped["units_sold"].apply(_compute_lags)
    frame["lag_1"] = lags["lag_1"].values
    frame["lag_2"] = lags["lag_2"].values
    rolling = grouped["units_sold"].apply(_compute_rolling)
    frame["rolling_mean_4"] = rolling["rolling_mean_4"].values
    frame["rolling_std_4"] = rolling["rolling_std_4"].values
    frame["momentum"] = frame["lag_1"] - frame["lag_2"]
    return _fill_and_validate_lags(frame)
