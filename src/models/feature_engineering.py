"""Feature engineering for the demand forecaster.

Pre-computed features in `weekly_df_final_for_modeling.csv` are reused as-is.
Derived features are added in pure functions.
"""
from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from src.utils.helpers import safe_divide
from src.utils.logger import get_logger

logger = get_logger(__name__)

LIFECYCLE_ENCODING: Final[dict[str, int]] = {"Decline": 0, "Mature": 1, "Growth": 2}
CHANNEL_ENCODING: Final[dict[str, int]] = {"Discount": 0, "Retail": 1, "E-commerce": 2}
REGION_ENCODING: Final[dict[str, int]] = {"PL-South": 0, "PL-Central": 1, "PL-North": 2}

PRECOMPUTED_FEATURES: Final[list[str]] = [
    "lag_1", "lag_2", "rolling_mean_4", "rolling_std_4", "momentum",
    "is_holiday_peak", "is_holiday_week", "is_summer", "is_winter",
    "week_number", "month", "year", "sku_age",
    "promotion_flag", "price_unit", "delivery_days", "stock_available",
]

DERIVED_FEATURES: Final[list[str]] = [
    "lifecycle_encoded", "channel_encoded", "region_encoded",
    "price_vs_category_avg", "stock_to_demand_ratio", "promo_lag_1",
]

ALL_FEATURES: Final[list[str]] = PRECOMPUTED_FEATURES + DERIVED_FEATURES


def add_derived_features(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """Append derived features required by the forecaster.

    Args:
        weekly_df: Weekly DataFrame with `category` joinable from `products`.

    Returns:
        New DataFrame with derived columns added.
    """
    frame = weekly_df.copy()
    frame["lifecycle_encoded"] = frame["lifecycle_stage"].map(LIFECYCLE_ENCODING).fillna(1).astype(int)
    frame["channel_encoded"] = frame["channel"].map(CHANNEL_ENCODING).fillna(1).astype(int)
    frame["region_encoded"] = frame["region"].map(REGION_ENCODING).fillna(1).astype(int)
    if "category" in frame.columns:
        category_avg = frame.groupby("category")["price_unit"].transform("mean")
        safe_avg = category_avg.replace(0, np.nan)
    else:
        scalar_avg = float(frame["price_unit"].mean())
        # WHY: at inference time `recent_data` lacks the `category` column
        # (it lives in the products table); fall back to series-level mean.
        safe_avg = scalar_avg if scalar_avg != 0 else np.nan
    frame["price_vs_category_avg"] = frame["price_unit"] / safe_avg
    frame["price_vs_category_avg"] = frame["price_vs_category_avg"].fillna(1.0)
    frame["stock_to_demand_ratio"] = frame.apply(
        lambda row: safe_divide(row.get("stock_available", 0.0), row.get("rolling_mean_4", 0.0), 0.0),
        axis=1,
    )
    grouped = frame.sort_values(["sku", "channel", "region", "week"]).groupby(
        ["sku", "channel", "region"], group_keys=False
    )
    frame["promo_lag_1"] = grouped["promotion_flag"].shift(1).fillna(0).astype(int)
    return frame


def build_training_features(weekly_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build the (X, y) training matrix.

    Args:
        weekly_df: Weekly DataFrame including `target_next_week`.

    Returns:
        Tuple of (feature matrix, target series). Rows missing the target
        are dropped; lag NaNs are forward-filled within each (sku, channel, region).

    Raises:
        ValueError: If `target_next_week` column is missing.
    """
    if "target_next_week" not in weekly_df.columns:
        raise ValueError("weekly_df missing required 'target_next_week' column.")
    enriched = add_derived_features(weekly_df)
    fill_columns = [col for col in PRECOMPUTED_FEATURES if col in enriched.columns]
    enriched[fill_columns] = enriched.groupby(["sku", "channel", "region"])[fill_columns].transform(
        lambda series: series.ffill().bfill()
    )
    enriched[fill_columns] = enriched[fill_columns].fillna(0.0)
    available = [col for col in ALL_FEATURES if col in enriched.columns]
    feature_frame = enriched[available].copy()
    target = enriched["target_next_week"].copy()
    valid_mask = target.notna()
    feature_frame = feature_frame.loc[valid_mask].reset_index(drop=True)
    target = target.loc[valid_mask].reset_index(drop=True)
    feature_frame.replace({np.inf: 0.0, -np.inf: 0.0}, inplace=True)
    feature_frame.fillna(0.0, inplace=True)
    return feature_frame, target


def build_inference_features(
    sku: str,
    channel: str,
    region: str,
    recent_data: pd.DataFrame,
) -> pd.DataFrame:
    """Build a single-row inference DataFrame from recent history.

    Args:
        sku: SKU identifier.
        channel: Channel name.
        region: Region name.
        recent_data: Weekly history filtered to (sku, channel, region) and sorted.

    Returns:
        One-row DataFrame ready for `model.predict`.

    Raises:
        ValueError: If `recent_data` is empty.
    """
    if recent_data.empty:
        raise ValueError(f"no recent data for {sku=} {channel=} {region=}")
    enriched = add_derived_features(recent_data)
    last = enriched.iloc[[-1]].copy()
    last["sku"] = sku
    last["channel"] = channel
    last["region"] = region
    available = [col for col in ALL_FEATURES if col in last.columns]
    feature_frame = last[available].copy()
    feature_frame.replace({np.inf: 0.0, -np.inf: 0.0}, inplace=True)
    feature_frame.fillna(0.0, inplace=True)
    return feature_frame


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return the feature columns present in `frame` in canonical order."""
    return [col for col in ALL_FEATURES if col in frame.columns]
