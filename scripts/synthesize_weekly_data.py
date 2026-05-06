"""Synthetic future week generator for weekly_features.

Reads historical distribution stats per (sku, channel, region) from the DB,
then generates N synthetic future weeks preserving realistic ranges.

CLI:
    python -m scripts.synthesize_weekly_data --weeks 4 [--seed 42] [--insert]

Without --insert: prints summary stats and first 5 rows.
With    --insert: upserts into weekly_features (idempotent — never overwrites).
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from sqlalchemy import text

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.database.connection import session_scope
from src.etl.loaders import upsert_dataframe
from src.utils.logger import get_logger

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout)
logger = get_logger(__name__)

_WEEK_DELTA: Final[timedelta] = timedelta(weeks=1)
_NOISE_FRACTION: Final[float] = 0.05  # ±5% price noise
_STD_FACTOR: Final[float] = 0.5       # WHY: halved std keeps synthetic spread realistic but narrower
_CLIP_LOW: Final[float] = 0.5         # WHY: allow down to 50% of historical min (preserves negatives)
_CLIP_HIGH: Final[float] = 1.5        # WHY: cap at 150% of historical max to avoid implausible spikes
_HOLIDAY_WEEKS: Final[frozenset[int]] = frozenset({14, 17, 18, 43, 44, 51, 52})
_SUMMER_MONTHS: Final[frozenset[int]] = frozenset({6, 7, 8})
_WINTER_MONTHS: Final[frozenset[int]] = frozenset({12, 1, 2})


def _load_series_stats(session) -> pd.DataFrame:  # type: ignore[type-arg]
    """Load per-(sku,channel,region) distribution stats and latest metadata."""
    sql = text("""
        SELECT wf.sku, wf.channel, wf.region,
               AVG(wf.units_sold)          AS mean_us,
               STDDEV(wf.units_sold)       AS std_us,
               MIN(wf.units_sold)          AS min_us,
               MAX(wf.units_sold)          AS max_us,
               AVG(wf.price_unit)          AS mean_price,
               AVG(wf.stock_available)     AS mean_stock,
               AVG(wf.delivery_days)       AS mean_dd,
               MAX(wf.sku_age)             AS max_sku_age,
               MAX(wf.week)                AS latest_week
        FROM weekly_features wf
        GROUP BY wf.sku, wf.channel, wf.region
    """)
    df = pd.DataFrame(session.execute(sql).fetchall(),
                      columns=["sku", "channel", "region", "mean_us", "std_us",
                               "min_us", "max_us", "mean_price", "mean_stock",
                               "mean_dd", "max_sku_age", "latest_week"])
    # WHY: psycopg2 returns NUMERIC as Decimal; cast to float for numpy arithmetic
    numeric_cols = ["mean_us", "std_us", "min_us", "max_us", "mean_price", "mean_stock", "mean_dd"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    df["max_sku_age"] = df["max_sku_age"].astype(int)
    return df


def _load_lifecycle(session) -> pd.DataFrame:  # type: ignore[type-arg]
    """Load latest lifecycle_stage per SKU."""
    sql = text("""
        SELECT DISTINCT ON (sku) sku, lifecycle_stage
        FROM weekly_features ORDER BY sku, week DESC
    """)
    return pd.DataFrame(session.execute(sql).fetchall(), columns=["sku", "lifecycle_stage"])


def _load_seasonality(session) -> pd.DataFrame:  # type: ignore[type-arg]
    """Compute per-(sku,channel,region,month) seasonality factor."""
    sql = text("""
        SELECT sku, channel, region, month,
               AVG(units_sold) AS month_mean
        FROM weekly_features
        GROUP BY sku, channel, region, month
    """)
    monthly = pd.DataFrame(session.execute(sql).fetchall(),
                           columns=["sku", "channel", "region", "month", "month_mean"])
    # WHY: psycopg2 returns NUMERIC as Decimal; cast before arithmetic
    monthly["month_mean"] = monthly["month_mean"].astype(float)
    monthly["month"] = monthly["month"].astype(int)
    overall = monthly.groupby(["sku", "channel", "region"])["month_mean"].mean().reset_index()
    overall.rename(columns={"month_mean": "overall_mean"}, inplace=True)
    merged = monthly.merge(overall, on=["sku", "channel", "region"])
    # WHY: factor > 1 → that month is above-average demand; used to scale samples
    safe_overall = merged["overall_mean"].replace(0.0, np.nan)
    merged["seasonality_factor"] = (merged["month_mean"] / safe_overall).fillna(1.0)
    return merged[["sku", "channel", "region", "month", "seasonality_factor"]]


def _latest_db_week(session) -> date:  # type: ignore[type-arg]
    """Return the latest week date stored in weekly_features."""
    result = session.execute(text("SELECT MAX(week) FROM weekly_features")).scalar()
    return result  # type: ignore[return-value]


def _calendar_flags(target_date: date) -> dict:
    """Derive calendar metadata for a synthetic week."""
    wn = int(target_date.strftime("%V"))
    m = target_date.month
    return {
        "week_number": wn,
        "month": m,
        "year": target_date.year,
        "is_holiday_week": int(wn in _HOLIDAY_WEEKS),
        "is_holiday_peak": int(wn in _HOLIDAY_WEEKS),
        "is_summer": int(m in _SUMMER_MONTHS),
        "is_winter": int(m in _WINTER_MONTHS),
    }


def _build_synthetic_rows(
    stats: pd.DataFrame,
    lifecycle_df: pd.DataFrame,
    seasonality_df: pd.DataFrame,
    start_week: date,
    n_weeks: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate synthetic weekly rows for every (sku, channel, region) series."""
    lifecycle_map = dict(zip(lifecycle_df["sku"], lifecycle_df["lifecycle_stage"]))
    season_index = seasonality_df.set_index(["sku", "channel", "region", "month"])["seasonality_factor"]
    all_rows: list[dict] = []
    for _, stat in stats.iterrows():
        sku, ch, reg = stat["sku"], stat["channel"], stat["region"]
        mean_us = float(stat["mean_us"])
        std_us = float(stat["std_us"] or 0.0)
        min_us = float(stat["min_us"])
        max_us = float(stat["max_us"])
        # Pass 1: generate all synthetic units_sold values first
        units_series: list[float] = []
        for step in range(n_weeks):
            target_date = start_week + _WEEK_DELTA * step
            m = target_date.month
            sf = float(season_index.get((sku, ch, reg, m), 1.0))
            # WHY: Normal(mean*sf, std*0.5) gives seasonal variance with tighter spread
            raw = rng.normal(mean_us * sf, std_us * _STD_FACTOR)
            # WHY: preserve negatives by allowing clip_low < 0 when min_us < 0
            units_series.append(float(np.clip(raw, min_us * _CLIP_LOW, max_us * _CLIP_HIGH)))
        # Pass 2: build rows with correct lag/rolling features and target_next_week
        for step in range(n_weeks):
            target_date = start_week + _WEEK_DELTA * step
            flags = _calendar_flags(target_date)
            units = units_series[step]
            prev = units_series[:step]  # synthetic units produced before this step
            lag_1 = prev[-1] if prev else mean_us
            lag_2 = prev[-2] if len(prev) >= 2 else mean_us
            window = (prev[-3:] if len(prev) >= 3 else [mean_us] * (3 - len(prev)) + prev) + [units]
            rolling_mean = float(np.mean(window[-4:]))
            rolling_std = float(np.std(window[-4:]) if len(window) >= 2 else 0.0)
            # WHY: last synthetic row uses series mean as target_next_week placeholder;
            # no true future observation exists — caller must treat this as approximate.
            target_next = units_series[step + 1] if step + 1 < n_weeks else mean_us
            price = float(rng.uniform(
                float(stat["mean_price"]) * (1 - _NOISE_FRACTION),
                float(stat["mean_price"]) * (1 + _NOISE_FRACTION),
            ))
            promo_flag = int(rng.binomial(1, 0.3))  # WHY: ~30% promo rate mirrors training data
            row = {
                "sku": sku, "channel": ch, "region": reg,
                "week": target_date, "units_sold": units,
                "stock_available": float(stat["mean_stock"]),
                "promotion_flag": promo_flag,
                "price_unit": price,
                "delivery_days": float(stat["mean_dd"]),
                "lag_1": lag_1, "lag_2": lag_2,
                "rolling_mean_4": rolling_mean, "rolling_std_4": rolling_std,
                "momentum": lag_1 - lag_2,
                "target_next_week": target_next,
                "sku_age": int(stat["max_sku_age"]) + step + 1,
                "lifecycle_stage": lifecycle_map.get(sku, "Mature"),
                **flags,
            }
            all_rows.append(row)
    return pd.DataFrame(all_rows)


def synthesize(n_weeks: int = 1, seed: int | None = None, insert: bool = False) -> pd.DataFrame:
    """Generate synthetic weekly rows and optionally persist them.

    Args:
        n_weeks: Number of future weeks to generate per series.
        seed: Optional RNG seed for reproducibility.
        insert: If True, upsert rows into weekly_features.

    Returns:
        DataFrame of synthetic rows.
    """
    rng = np.random.default_rng(seed)
    with session_scope() as session:
        latest_week = _latest_db_week(session)
        stats = _load_series_stats(session)
        lifecycle_df = _load_lifecycle(session)
        seasonality_df = _load_seasonality(session)
    start_week = latest_week + _WEEK_DELTA
    logger.info("synthesis_start", extra={
        "start_week": str(start_week), "n_weeks": n_weeks, "series": len(stats),
    })
    synthetic_df = _build_synthetic_rows(stats, lifecycle_df, seasonality_df,
                                         start_week, n_weeks, rng)
    logger.info("synthesis_complete", extra={"rows": len(synthetic_df)})
    if insert:
        upsert_dataframe(
            "weekly_features", synthetic_df,
            conflict_cols=["sku", "week", "channel", "region"],
        )
        logger.info("synthetic_rows_inserted", extra={"rows": len(synthetic_df)})
    return synthetic_df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic weekly demand data.")
    parser.add_argument("--weeks", type=int, default=1, help="Number of future weeks to generate.")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed.")
    parser.add_argument("--insert", action="store_true", help="Write rows to weekly_features table.")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = _parse_args()
    result = synthesize(n_weeks=args.weeks, seed=args.seed, insert=args.insert)
    print(f"\nSynthesized {len(result)} rows ({args.weeks} weeks × {len(result) // args.weeks} series)")
    print("\nSummary stats:")
    print(result[["units_sold", "price_unit", "stock_available"]].describe().round(2).to_string())
    print("\nFirst 5 rows:")
    print(result[["sku", "channel", "region", "week", "units_sold", "price_unit"]].head().to_string(index=False))


if __name__ == "__main__":
    main()
