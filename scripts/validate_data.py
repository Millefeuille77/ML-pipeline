"""Pre-training data validation gate.

Loads weekly_features from DB and runs schema, cardinality, missingness,
distribution, and recency checks. Returns exit code 0 (pass) or 1 (fail).

CLI:
    python -m scripts.validate_data [--allow-stale]

Also exposes a programmatic interface via `validate()` returning ValidationReport.
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Final

import pandas as pd
from sqlalchemy import text

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.database.connection import session_scope
from src.utils.logger import get_logger

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout)
logger = get_logger(__name__)

# WHY: columns the ML pipeline requires to be present and numeric/typed
_REQUIRED_COLUMNS: Final[list[str]] = [
    "sku", "week", "channel", "region", "units_sold", "stock_available",
    "promotion_flag", "price_unit", "delivery_days",
    "is_holiday_peak", "week_number", "month", "year",
    "is_holiday_week", "is_summer", "is_winter", "sku_age",
    "lifecycle_stage", "lag_1", "lag_2", "rolling_mean_4",
    "rolling_std_4", "momentum", "target_next_week",
]
_NUMERIC_COLUMNS: Final[list[str]] = [
    "units_sold", "stock_available", "price_unit", "delivery_days",
    "is_holiday_peak", "week_number", "month", "year", "is_holiday_week",
    "is_summer", "is_winter", "sku_age", "lag_1", "lag_2",
    "rolling_mean_4", "rolling_std_4", "momentum", "target_next_week",
]
# WHY: lag columns are legitimately NaN at series start — exclude from missingness check
_LAG_COLUMNS: Final[frozenset[str]] = frozenset({"lag_1", "lag_2"})
_MAX_MISSING_RATE: Final[float] = 0.01  # 1% threshold for non-lag columns

# Distribution sanity bounds from CLAUDE.md §5 and observed training-data stats
_UNITS_SOLD_MEAN_LOW: Final[float] = 50.0
_UNITS_SOLD_MEAN_HIGH: Final[float] = 200.0
_PRICE_MEAN_LOW: Final[float] = 3.0
_PRICE_MEAN_HIGH: Final[float] = 8.0

_EXPECTED_SKUS: Final[int] = 30
_EXPECTED_CHANNELS: Final[int] = 3
_EXPECTED_REGIONS: Final[int] = 3
_RECENCY_DAYS: Final[int] = 30


@dataclass
class ValidationReport:
    """Result of a full validation pass.

    Attributes:
        checks: List of check result dicts with keys: name, level, message.
        passed: False if any FAIL-level check fired.
    """

    checks: list[dict] = field(default_factory=list)
    passed: bool = True


def _check(report: ValidationReport, name: str, ok: bool, level: str, msg: str) -> None:
    """Append a check result and mark report failed on FAIL level."""
    report.checks.append({"name": name, "level": level, "ok": ok, "message": msg})
    if not ok and level == "FAIL":
        report.passed = False
    log_fn = logger.warning if not ok else logger.info
    # WHY: "msg" and "name" are reserved LogRecord fields; use check_msg to avoid KeyError
    log_fn("validation_check", extra={"check": name, "ok": ok, "check_level": level, "check_msg": msg})


def _run_schema_check(df: pd.DataFrame, report: ValidationReport) -> None:
    """Verify all expected columns are present with compatible dtypes."""
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    _check(report, "schema_columns", not missing, "FAIL",
           f"Missing columns: {missing}" if missing else "All 24 required columns present")
    for col in _NUMERIC_COLUMNS:
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            _check(report, f"dtype_{col}", False, "FAIL", f"Column {col!r} should be numeric")


def _run_cardinality_check(df: pd.DataFrame, report: ValidationReport) -> None:
    """Verify expected distinct counts for SKUs, channels, regions."""
    skus = df["sku"].nunique() if "sku" in df.columns else 0
    channels = df["channel"].nunique() if "channel" in df.columns else 0
    regions = df["region"].nunique() if "region" in df.columns else 0
    _check(report, "cardinality_skus", skus == _EXPECTED_SKUS, "WARN",
           f"SKUs: {skus} (expected {_EXPECTED_SKUS})")
    _check(report, "cardinality_channels", channels == _EXPECTED_CHANNELS, "WARN",
           f"Channels: {channels} (expected {_EXPECTED_CHANNELS})")
    _check(report, "cardinality_regions", regions == _EXPECTED_REGIONS, "WARN",
           f"Regions: {regions} (expected {_EXPECTED_REGIONS})")


def _run_missingness_check(df: pd.DataFrame, report: ValidationReport) -> None:
    """Check non-lag columns stay under 1% missing."""
    for col in _REQUIRED_COLUMNS:
        if col not in df.columns or col in _LAG_COLUMNS:
            continue
        rate = df[col].isna().mean()
        ok = rate <= _MAX_MISSING_RATE
        _check(report, f"missing_{col}", ok, "FAIL",
               f"{col}: {rate:.2%} NaN (limit {_MAX_MISSING_RATE:.0%})")


def _run_distribution_check(df: pd.DataFrame, report: ValidationReport) -> None:
    """Warn when units_sold or price_unit mean fall outside historical bounds."""
    if "units_sold" in df.columns:
        m = float(df["units_sold"].mean())
        ok = _UNITS_SOLD_MEAN_LOW <= m <= _UNITS_SOLD_MEAN_HIGH
        _check(report, "dist_units_sold", ok, "WARN",
               f"units_sold mean={m:.2f} (expected [{_UNITS_SOLD_MEAN_LOW},{_UNITS_SOLD_MEAN_HIGH}])")
    if "price_unit" in df.columns:
        m = float(df["price_unit"].mean())
        ok = _PRICE_MEAN_LOW <= m <= _PRICE_MEAN_HIGH
        _check(report, "dist_price_unit", ok, "WARN",
               f"price_unit mean={m:.4f} (expected [{_PRICE_MEAN_LOW},{_PRICE_MEAN_HIGH}])")


def _run_recency_check(df: pd.DataFrame, report: ValidationReport, allow_stale: bool) -> None:
    """Warn if the latest week is older than 30 days."""
    if "week" not in df.columns:
        return
    latest = pd.to_datetime(df["week"]).max().date()
    days_ago = (date.today() - latest).days
    ok = days_ago <= _RECENCY_DAYS or allow_stale
    _check(report, "recency", ok, "WARN",
           f"Latest week {latest} is {days_ago} days ago (--allow-stale={'set' if allow_stale else 'not set'})")


def _load_weekly_features() -> pd.DataFrame:
    """Load all rows from weekly_features into a DataFrame with correct dtypes."""
    with session_scope() as session:
        rows = session.execute(text(
            "SELECT sku, week, channel, region, units_sold, stock_available, "
            "promotion_flag, price_unit, delivery_days, is_holiday_peak, "
            "week_number, month, year, is_holiday_week, is_summer, is_winter, "
            "sku_age, lifecycle_stage, lag_1, lag_2, rolling_mean_4, "
            "rolling_std_4, momentum, target_next_week FROM weekly_features"
        )).fetchall()
    columns = [
        "sku", "week", "channel", "region", "units_sold", "stock_available",
        "promotion_flag", "price_unit", "delivery_days", "is_holiday_peak",
        "week_number", "month", "year", "is_holiday_week", "is_summer", "is_winter",
        "sku_age", "lifecycle_stage", "lag_1", "lag_2", "rolling_mean_4",
        "rolling_std_4", "momentum", "target_next_week",
    ]
    df = pd.DataFrame(rows, columns=columns)
    # WHY: psycopg2 returns NUMERIC columns as Decimal objects → object dtype;
    # cast explicitly so pandas.api.types.is_numeric_dtype returns True.
    for col in _NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def validate(allow_stale: bool = False) -> ValidationReport:
    """Run all validation checks against the current weekly_features table.

    Args:
        allow_stale: If True, suppresses the recency WARN for old data.

    Returns:
        ValidationReport with per-check results and overall pass/fail.
    """
    report = ValidationReport()
    df = _load_weekly_features()
    logger.info("validation_loaded", extra={"rows": len(df)})
    _run_schema_check(df, report)
    _run_cardinality_check(df, report)
    _run_missingness_check(df, report)
    _run_distribution_check(df, report)
    _run_recency_check(df, report, allow_stale)
    status = "PASS" if report.passed else "FAIL"
    logger.info("validation_complete", extra={"status": status, "checks": len(report.checks)})
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate weekly_features before retraining.")
    parser.add_argument("--allow-stale", action="store_true",
                        help="Do not warn about data older than 30 days.")
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint. Returns exit code 0 (pass) or 1 (fail)."""
    args = _parse_args()
    report = validate(allow_stale=args.allow_stale)
    fails = [c for c in report.checks if not c["ok"] and c["level"] == "FAIL"]
    warns = [c for c in report.checks if not c["ok"] and c["level"] == "WARN"]
    print(f"\nValidation: {'PASS' if report.passed else 'FAIL'} "
          f"({len(fails)} failures, {len(warns)} warnings)\n")
    for check in report.checks:
        icon = "OK" if check["ok"] else check["level"]
        print(f"  [{icon:4s}] {check['name']}: {check['message']}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
