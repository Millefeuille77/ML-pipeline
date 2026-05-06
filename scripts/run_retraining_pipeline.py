"""MLOps retraining pipeline with data validation, drift detection, and lineage.

For each of 5 categories: validate data → detect drift → train challenger →
compare to incumbent → promote or reject → write model_lineage audit row.

CLI:
    python -m scripts.run_retraining_pipeline [--dry-run]

--dry-run: prints all decisions without saving models or writing to DB.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Final

import pandas as pd
from sqlalchemy import text

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import get_settings
from scripts.validate_data import validate
from src.database.connection import session_scope
from src.models.evaluation import compare_models
from src.models.feature_engineering import build_training_features
from src.models.forecaster import train_gbr, train_ridge
from src.models.registry import compare_to_incumbent, record_lineage, save_model
from src.utils.logger import get_logger

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout)
logger = get_logger(__name__)

_CATEGORY_PREFIX: Final[dict[str, str]] = {
    "MI": "Milk", "YO": "Yogurt", "RE": "ReadyMeal", "JU": "Juice", "SN": "SnackBar",
}
_MIN_TRAIN_WEEKS: Final[int] = 100
_TEST_WEEKS: Final[int] = 4
_N_SPLITS: Final[int] = 5
_DRIFT_SIGMA: Final[float] = 2.0   # WHY: 2σ flag is standard SPC rule for detecting process shifts


_NUMERIC_COLS: Final[list[str]] = [
    "units_sold", "stock_available", "price_unit", "delivery_days",
    "is_holiday_peak", "week_number", "month", "year", "is_holiday_week",
    "is_summer", "is_winter", "sku_age", "lag_1", "lag_2",
    "rolling_mean_4", "rolling_std_4", "momentum", "target_next_week", "promotion_flag",
]


def _load_weekly_features() -> pd.DataFrame:
    """Load weekly_features from DB with correct numeric dtypes."""
    with session_scope() as session:
        rows = session.execute(text(
            "SELECT wf.sku, wf.week, wf.channel, wf.region, wf.units_sold, "
            "wf.stock_available, wf.promotion_flag, wf.price_unit, wf.delivery_days, "
            "wf.is_holiday_peak, wf.week_number, wf.month, wf.year, wf.is_holiday_week, "
            "wf.is_summer, wf.is_winter, wf.sku_age, wf.lifecycle_stage, wf.lag_1, "
            "wf.lag_2, wf.rolling_mean_4, wf.rolling_std_4, wf.momentum, "
            "wf.target_next_week, p.category "
            "FROM weekly_features wf "
            "JOIN products p ON wf.sku = p.sku"
        )).fetchall()
    columns = [
        "sku", "week", "channel", "region", "units_sold", "stock_available",
        "promotion_flag", "price_unit", "delivery_days", "is_holiday_peak",
        "week_number", "month", "year", "is_holiday_week", "is_summer", "is_winter",
        "sku_age", "lifecycle_stage", "lag_1", "lag_2", "rolling_mean_4",
        "rolling_std_4", "momentum", "target_next_week", "category",
    ]
    df = pd.DataFrame(rows, columns=columns)
    # WHY: psycopg2 returns NUMERIC columns as Decimal; cast to float for sklearn arithmetic
    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["week"] = pd.to_datetime(df["week"])
    logger.info("training_data_loaded", extra={"rows": len(df), "categories": df["category"].nunique()})
    return df


def _compute_data_hash(df: pd.DataFrame) -> str:
    """Return SHA-256 hex of sorted training rows for reproducible data fingerprinting."""
    # WHY: deterministic sort ensures same data yields same hash regardless of DB retrieval order
    content = df.sort_values(["sku", "week", "channel", "region"]).to_csv(index=False).encode()
    return hashlib.sha256(content).hexdigest()[:32]


def _load_incumbent_baselines(category: str) -> dict[str, dict[str, float]]:
    """Load feature_baselines from the latest incumbent's metadata sidecar.

    Returns empty dict if no incumbent exists or baselines not stored.
    """
    model_dir = get_settings().resolved_model_dir()
    name = f"forecaster_{category.lower().replace(' ', '_')}"
    # WHY: glob for latest version rather than calling get_model to avoid loading the estimator
    pattern = f"{name}_v*.json"
    candidates = sorted(model_dir.glob(pattern))
    if not candidates:
        return {}
    try:
        meta = json.loads(candidates[-1].read_text(encoding="utf-8"))
        return meta.get("feature_baselines", {})
    except (json.JSONDecodeError, OSError):
        logger.warning("baseline_load_failed", extra={"category": category})
        return {}


def _detect_drift(
    features_df: pd.DataFrame,
    baselines: dict[str, dict[str, float]],
) -> list[dict]:
    """Compare current feature means against stored baselines.

    Args:
        features_df: Current training feature matrix (X only, numeric).
        baselines: Mapping feature -> {mean, std} from the incumbent metadata.

    Returns:
        List of drift warning dicts for features exceeding 2σ shift.
    """
    if not baselines:
        return []
    warnings: list[dict] = []
    for feat, baseline in baselines.items():
        if feat not in features_df.columns:
            continue
        current_mean = float(features_df[feat].mean())
        b_mean = float(baseline.get("mean", current_mean))
        b_std = float(baseline.get("std", 0.0))
        if b_std < 1e-9:
            continue
        shift = abs(current_mean - b_mean)
        if shift > _DRIFT_SIGMA * b_std:
            warnings.append({
                "feature": feat,
                "current_mean": round(current_mean, 4),
                "baseline_mean": round(b_mean, 4),
                "baseline_std": round(b_std, 4),
                "sigma_shift": round(shift / b_std, 2),
                "severity": "warn",
            })
    if warnings:
        logger.warning("drift_detected", extra={"count": len(warnings),
                       "features": [w["feature"] for w in warnings]})
    return warnings


def _safe_n_splits(unique_weeks: int) -> int:
    """Clamp n_splits to what the data can support."""
    max_splits = (unique_weeks - _MIN_TRAIN_WEEKS) // _TEST_WEEKS
    return max(1, min(_N_SPLITS, max_splits))


def _train_category(
    category: str, subset: pd.DataFrame, dry_run: bool, run_id: str,
) -> dict:
    """Run comparison, select winner, handle incumbent comparison, write lineage."""
    t0 = time.perf_counter()
    # WHY: use only the historical (non-synthetic) slice for validation splits if possible;
    # full dataset used so synthetic weeks help robustness without data leakage in CV
    unique_weeks = int(subset["week"].nunique())
    n_splits = _safe_n_splits(unique_weeks)
    logger.info("category_comparison_start", extra={"category": category, "rows": len(subset), "splits": n_splits})
    comparison = compare_models(subset, n_splits=n_splits,
                                train_weeks=_MIN_TRAIN_WEEKS, test_weeks=_TEST_WEEKS)
    features_df, target = build_training_features(subset)
    bundle = train_gbr(features_df, target) if comparison.winner == "gbr" else train_ridge(features_df, target)
    bundle.category = category
    data_hash = _compute_data_hash(subset)
    baselines = _load_incumbent_baselines(category)
    drift_warnings = _detect_drift(features_df, baselines)
    candidate_metrics = (comparison.gbr_metrics if comparison.winner == "gbr"
                         else comparison.ridge_metrics)
    model_name = f"forecaster_{category.lower().replace(' ', '_')}"
    decision = compare_to_incumbent(model_name, candidate_metrics)
    incumbent_version: str | None = None
    incumbent_mape: float | None = None
    # WHY: lineage rows record incumbent metrics even on rejection so the audit
    # trail captures what the challenger was being compared against.
    try:
        from src.models.registry import list_models as _list_models
        incumbents = _list_models(model_name)
        if incumbents:
            incumbent_version = f"{model_name}_v{incumbents[-1].version}"
            incumbent_mape = incumbents[-1].metrics.get("mape")
    except (FileNotFoundError, AttributeError, KeyError) as exc:
        logger.warning(
            "incumbent_lookup_failed",
            extra={"category": category, "model_name": model_name, "error": str(exc)},
        )
    saved_version = f"{model_name}_vPENDING"
    if decision in ("promoted", "no_incumbent") and not dry_run:
        info = save_model(
            bundle, name=model_name, metrics=candidate_metrics,
            features=bundle.feature_names, training_rows=len(features_df),
            category=category, feature_frame=features_df,
        )
        saved_version = f"{model_name}_v{info.version}"
        logger.info("model_promoted", extra={"category": category, "version": info.version})
    elif dry_run:
        saved_version = f"{model_name}_v[dry-run]"
        logger.info("dry_run_skip_save", extra={"category": category, "decision": decision})
    else:
        logger.info("model_rejected_no_save", extra={"category": category})
    if not dry_run:
        record_lineage(
            run_id=f"{run_id}_{category.lower()[:4]}",
            category=category,
            model_version=saved_version,
            data_hash=data_hash,
            training_rows=len(features_df),
            train_metrics=candidate_metrics,
            incumbent_version=incumbent_version,
            incumbent_mape=incumbent_mape,
            promotion_decision=decision,
            drift_warnings=drift_warnings,
            notes=comparison.rationale,
        )
    elapsed = time.perf_counter() - t0
    return {
        "category": category,
        "train_rows": len(features_df),
        "winner": comparison.winner,
        "mape": candidate_metrics.get("mape", 0.0),
        "rmse": candidate_metrics.get("rmse", 0.0),
        "r2": candidate_metrics.get("r2", 0.0),
        "decision": decision,
        "drift_count": len(drift_warnings),
        "elapsed_s": elapsed,
    }


def _print_summary(rows: list[dict]) -> None:
    """Print markdown evaluation table with promotion decisions."""
    print("\n| Category | Train rows | Model | MAPE | RMSE | R² | Decision | Drift |")
    print("|---|---|---|---|---|---|---|---|")
    for row in sorted(rows, key=lambda r: r["category"]):
        drifts = f"{row['drift_count']} warn" if row["drift_count"] else "clean"
        print(
            f"| {row['category']} | {row['train_rows']} | {row['winner']} "
            f"| {row['mape']:.1f}% | {row['rmse']:.1f} | {row['r2']:.2f} "
            f"| {row['decision']} | {drifts} |"
        )


def run_pipeline(dry_run: bool = False) -> list[dict]:
    """Full retraining pipeline.

    Args:
        dry_run: If True, skip registry writes and DB lineage inserts.

    Returns:
        Summary rows per category.
    """
    run_id = uuid.uuid4().hex[:16]
    logger.info("pipeline_start", extra={"run_id": run_id, "dry_run": dry_run})
    validation_report = validate(allow_stale=True)
    if not validation_report.passed:
        fails = [c["message"] for c in validation_report.checks if not c["ok"] and c["level"] == "FAIL"]
        logger.error("pipeline_aborted_validation_failed", extra={"fails": fails})
        raise RuntimeError(f"Data validation failed: {fails}")
    logger.info("validation_passed", extra={"checks": len(validation_report.checks)})
    weekly_df = _load_weekly_features()
    summary_rows: list[dict] = []
    for category, subset in weekly_df.groupby("category"):
        category = str(category)
        try:
            row = _train_category(category, subset.copy(), dry_run=dry_run, run_id=run_id)
            summary_rows.append(row)
        except Exception as exc:
            logger.error("category_pipeline_failed", extra={"category": category, "error": str(exc)})
            raise
    logger.info("pipeline_complete", extra={"run_id": run_id, "categories": len(summary_rows)})
    return summary_rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MLOps retraining pipeline.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip registry saves and DB lineage writes; only print decisions.")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = _parse_args()
    if args.dry_run:
        print("\n[DRY RUN] No models will be saved and no lineage rows will be written.\n")
    rows = run_pipeline(dry_run=args.dry_run)
    _print_summary(rows)
    print()


if __name__ == "__main__":
    main()
