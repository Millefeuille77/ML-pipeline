"""Reusable EDA helper functions (used by routes and the EDA notebook)."""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from src.utils.helpers import safe_divide
from src.utils.logger import get_logger

logger = get_logger(__name__)

Period = Literal["weekly", "monthly"]


def _bucket_period(frame: pd.DataFrame, period: Period) -> pd.Series:
    """Return a Series of period-start dates for each row."""
    if "week" in frame.columns:
        as_dt = pd.to_datetime(frame["week"])
    elif "date" in frame.columns:
        as_dt = pd.to_datetime(frame["date"])
    else:
        raise ValueError("DataFrame missing 'week' or 'date' column.")
    if period == "monthly":
        return as_dt.dt.to_period("M").dt.start_time.dt.date
    return as_dt.dt.to_period("W-SUN").dt.start_time.dt.date


def sales_by_category(frame: pd.DataFrame, period: Period = "weekly") -> pd.DataFrame:
    """Aggregate units_sold by category and period."""
    bucket = _bucket_period(frame, period)
    grouped = frame.assign(period=bucket).groupby(["category", "period"], as_index=False)["units_sold"].sum()
    return grouped.sort_values(["category", "period"]).reset_index(drop=True)


def sales_by_channel(frame: pd.DataFrame, period: Period = "weekly") -> pd.DataFrame:
    """Aggregate units_sold by channel and period."""
    bucket = _bucket_period(frame, period)
    return (
        frame.assign(period=bucket)
        .groupby(["channel", "period"], as_index=False)["units_sold"]
        .sum()
        .sort_values(["channel", "period"])
        .reset_index(drop=True)
    )


def sales_by_region(frame: pd.DataFrame, period: Period = "weekly") -> pd.DataFrame:
    """Aggregate units_sold by region and period."""
    bucket = _bucket_period(frame, period)
    return (
        frame.assign(period=bucket)
        .groupby(["region", "period"], as_index=False)["units_sold"]
        .sum()
        .sort_values(["region", "period"])
        .reset_index(drop=True)
    )


def promo_impact_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute mean units sold with vs without promo, per SKU."""
    grouped = frame.groupby(["sku", "promotion_flag"], as_index=False)["units_sold"].mean()
    pivoted = grouped.pivot(index="sku", columns="promotion_flag", values="units_sold").reset_index()
    pivoted.columns.name = None
    pivoted = pivoted.rename(columns={0: "non_promo_avg", 1: "promo_avg"})
    pivoted["non_promo_avg"] = pivoted.get("non_promo_avg", 0.0).fillna(0.0)
    pivoted["promo_avg"] = pivoted.get("promo_avg", 0.0).fillna(0.0)
    pivoted["promo_lift_pct"] = pivoted.apply(
        lambda row: 100.0 * safe_divide(row["promo_avg"] - row["non_promo_avg"], row["non_promo_avg"], 0.0),
        axis=1,
    )
    return pivoted


def lifecycle_distribution(frame: pd.DataFrame) -> dict[str, int]:
    """Counts of SKUs in each Growth/Mature/Decline stage."""
    sku_stage = (
        frame.groupby("sku")["lifecycle_stage"]
        .agg(lambda series: series.mode().iloc[0] if not series.empty else "Mature")
    )
    counts = sku_stage.value_counts().to_dict()
    return {
        "Growth": int(counts.get("Growth", 0)),
        "Mature": int(counts.get("Mature", 0)),
        "Decline": int(counts.get("Decline", 0)),
    }


def seasonality_decomposition(sku: str, frame: pd.DataFrame) -> dict[str, float]:
    """Return monthly seasonality summary statistics for a single SKU.

    Args:
        sku: SKU identifier.
        frame: Weekly DataFrame.

    Returns:
        Dict of summary stats.
    """
    subset = frame.loc[frame["sku"] == sku]
    if subset.empty:
        return {"max_monthly": 0.0, "min_monthly": 0.0, "seasonality_strength": 1.0}
    monthly = subset.groupby("month")["units_sold"].mean()
    max_value = float(monthly.max() or 0.0)
    min_value = float(monthly.min() or 0.0)
    return {
        "max_monthly": max_value,
        "min_monthly": min_value,
        "seasonality_strength": float(safe_divide(max_value, min_value, 1.0)),
    }


def correlation_matrix(features_df: pd.DataFrame) -> pd.DataFrame:
    """Compute Pearson correlations across numeric columns."""
    numeric = features_df.select_dtypes(include=[np.number])
    return numeric.corr().fillna(0.0)


def top_products_by_units(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    """Top N SKUs by total units sold."""
    grouped = frame.groupby(["sku", "category"], as_index=False)["units_sold"].sum()
    return grouped.sort_values("units_sold", ascending=False).head(n).reset_index(drop=True)


def top_products_by_revenue(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    """Top N SKUs by total revenue (units_sold × price_unit)."""
    working = frame.copy()
    working["revenue"] = working["units_sold"].astype(float) * working["price_unit"].astype(float)
    grouped = working.groupby(["sku", "category"], as_index=False)["revenue"].sum()
    return grouped.sort_values("revenue", ascending=False).head(n).reset_index(drop=True)


def channel_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-channel summary including promo lift."""
    summary = frame.groupby("channel", as_index=False).agg(
        total_units_sold=("units_sold", "sum"),
        avg_price_unit=("price_unit", "mean"),
    )
    promo_means = (
        frame.groupby(["channel", "promotion_flag"])["units_sold"].mean().unstack(fill_value=0.0)
    )
    promo_means.columns = [f"flag_{int(col)}" for col in promo_means.columns]
    summary = summary.merge(promo_means, on="channel", how="left")
    summary["promo_lift_pct"] = summary.apply(
        lambda row: 100.0 * safe_divide(
            row.get("flag_1", 0.0) - row.get("flag_0", 0.0),
            row.get("flag_0", 0.0),
            0.0,
        ),
        axis=1,
    )
    return summary[["channel", "total_units_sold", "avg_price_unit", "promo_lift_pct"]]
