"""CLI entrypoint: train per-category forecasters, cluster, and emit a README table.

Usage:
    python scripts/train_all_models.py

Loads training data from CSV, trains GBR + Ridge per category via walk-forward
validation, selects the winning model, trains the K-Means clusterer, and saves
all artifacts through the registry.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Final

import pandas as pd

# WHY: ensure project root is on the path so `src.*` and `config.*` resolve
_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import get_settings
from src.models.clustering import fit_clustering
from src.models.evaluation import ComparisonReport, compare_models, walk_forward_validate
from src.models.feature_engineering import build_training_features
from src.models.forecaster import train_gbr, train_ridge
from src.models.registry import save_model
from src.utils.logger import get_logger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)
logger = get_logger(__name__)

_CATEGORY_PREFIX: Final[dict[str, str]] = {
    "MI": "Milk", "YO": "Yogurt", "RE": "ReadyMeal", "JU": "Juice", "SN": "SnackBar",
}
_MIN_TRAIN_WEEKS: Final[int] = 100
_TEST_WEEKS: Final[int] = 4
_N_SPLITS: Final[int] = 5


def _add_category(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """Derive `category` from the two-letter SKU prefix."""
    prefix = weekly_df["sku"].str[:2]
    weekly_df = weekly_df.copy()
    weekly_df["category"] = prefix.map(_CATEGORY_PREFIX)
    unmapped = weekly_df["category"].isna().sum()
    if unmapped:
        logger.warning("unmapped_skus", extra={"count": int(unmapped)})
    return weekly_df


def _safe_n_splits(unique_weeks: int, n_splits: int, train_weeks: int, test_weeks: int) -> int:
    """Return the largest valid n_splits for this category's week count.

    WHY: some categories have fewer weeks (e.g. SnackBar=117), so 5 splits
    at test_weeks=4 would need 116 weeks — barely feasible. Clamp gracefully
    rather than raising.
    """
    max_splits = (unique_weeks - train_weeks) // test_weeks
    if max_splits < n_splits:
        logger.warning(
            "reducing_splits",
            extra={"unique_weeks": unique_weeks, "requested": n_splits, "capped": max_splits},
        )
    return max(1, min(n_splits, max_splits))


def _run_comparison(category: str, subset: pd.DataFrame) -> ComparisonReport:
    """Walk-forward compare GBR vs Ridge for one category."""
    unique_weeks = int(subset["week"].nunique())
    splits = _safe_n_splits(unique_weeks, _N_SPLITS, _MIN_TRAIN_WEEKS, _TEST_WEEKS)
    logger.info("comparing_models", extra={"category": category, "splits": splits, "rows": len(subset)})
    return compare_models(subset, n_splits=splits, train_weeks=_MIN_TRAIN_WEEKS, test_weeks=_TEST_WEEKS)


def _train_winner(
    category: str, subset: pd.DataFrame, winner: str
) -> tuple[object, dict[str, float], list[str]]:
    """Train the winning model on all available data for final persistence."""
    features, target = build_training_features(subset)
    if winner == "ridge":
        bundle = train_ridge(features, target)
    else:
        bundle = train_gbr(features, target)
    bundle.category = category
    return bundle, bundle.metrics, bundle.feature_names


def _save_winner(
    category: str, bundle: object, winner: str,
    metrics: dict[str, float], features: list[str], training_rows: int,
) -> None:
    """Persist the winning model bundle via the registry."""
    name = f"forecaster_{category.lower().replace(' ', '_')}"
    save_model(
        bundle, name=name, metrics=metrics, features=features,
        training_rows=training_rows, category=category,
    )
    logger.info("winner_saved", extra={"category": category, "model": winner, "model_name": name})


def _train_forecasters(weekly_df: pd.DataFrame) -> list[dict]:
    """Train, compare, and persist one model per category. Returns summary rows."""
    rows: list[dict] = []
    for category, subset in weekly_df.groupby("category"):
        category = str(category)
        t0 = time.perf_counter()
        try:
            report = _run_comparison(category, subset)
            bundle, metrics, features = _train_winner(category, subset, report.winner)
            _save_winner(category, bundle, report.winner, metrics, features, len(subset))
            elapsed = time.perf_counter() - t0
            rows.append({
                "category": category,
                "train_rows": len(subset),
                "winner": report.winner,
                "mape": report.gbr_metrics["mape"] if report.winner == "gbr" else report.ridge_metrics["mape"],
                "rmse": report.gbr_metrics["rmse"] if report.winner == "gbr" else report.ridge_metrics["rmse"],
                "mae": report.gbr_metrics["mae"] if report.winner == "gbr" else report.ridge_metrics["mae"],
                "r2": report.gbr_metrics["r2"] if report.winner == "gbr" else report.ridge_metrics["r2"],
                "rationale": report.rationale,
                "elapsed_s": elapsed,
            })
            logger.info("category_done", extra={"category": category, "winner": report.winner, "elapsed_s": round(elapsed, 1)})
        except Exception as exc:
            logger.error("category_failed", extra={"category": category, "error": str(exc)})
            raise
    return rows


def _train_clusterer(weekly_df: pd.DataFrame) -> None:
    """Fit K-Means clustering on all 30 SKUs and persist via registry."""
    logger.info("clustering_start", extra={"skus": int(weekly_df["sku"].nunique())})
    t0 = time.perf_counter()
    cluster_model = fit_clustering(weekly_df)
    save_model(
        cluster_model, name="clusterer",
        metrics={}, features=[], training_rows=int(weekly_df["sku"].nunique()), category=None,
    )
    logger.info("clustering_done", extra={"elapsed_s": round(time.perf_counter() - t0, 1)})


def _print_table(summary_rows: list[dict]) -> None:
    """Print the README evaluation table to stdout."""
    header = "| Category | Train rows | Best model | MAPE | RMSE | MAE | R² |"
    sep =    "|---|---|---|---|---|---|---|"
    print(header)
    print(sep)
    for row in sorted(summary_rows, key=lambda r: r["category"]):
        print(
            f"| {row['category']} | {row['train_rows']} | {row['winner']} "
            f"| {row['mape']:.1f}% | {row['rmse']:.1f} | {row['mae']:.1f} | {row['r2']:.2f} |"
        )


def main() -> None:
    """Load CSV, train all models, emit README table."""
    settings = get_settings()
    csv_path = settings.resolved_raw_dir() / "weekly_df_final_for_modeling.csv"
    logger.info("loading_csv", extra={"path": str(csv_path)})
    weekly_df = pd.read_csv(csv_path, parse_dates=["week"])
    weekly_df = _add_category(weekly_df)
    logger.info("csv_loaded", extra={"rows": len(weekly_df), "categories": int(weekly_df["category"].nunique())})

    summary_rows = _train_forecasters(weekly_df)
    _train_clusterer(weekly_df)

    logger.info("all_training_complete")
    print("\n=== README Evaluation Table ===")
    _print_table(summary_rows)
    print()
    for row in sorted(summary_rows, key=lambda r: r["category"]):
        print(f"  {row['category']}: {row['rationale']} ({row['elapsed_s']:.1f}s)")


if __name__ == "__main__":
    main()
