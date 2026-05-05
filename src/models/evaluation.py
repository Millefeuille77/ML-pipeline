"""Walk-forward evaluation for time-series forecasting.

CRITICAL: only temporal splits — random shuffling on time-series is a
data-leakage bug.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from src.models.feature_engineering import build_training_features
from src.models.forecaster import (
    TrainedForecaster,
    evaluate_predictions,
    train_gbr,
    train_ridge,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_SPLITS: Final[int] = 5
_DEFAULT_TRAIN_WEEKS: Final[int] = 100
_DEFAULT_TEST_WEEKS: Final[int] = 4
_MAPE_TOLERANCE_PCT: Final[float] = 5.0


@dataclass
class FoldResult:
    """One walk-forward fold's metrics."""

    fold: int
    train_size: int
    test_size: int
    metrics: dict[str, float]


@dataclass
class EvaluationReport:
    """Aggregate walk-forward report."""

    folds: list[FoldResult]
    mean_metrics: dict[str, float]


@dataclass
class ComparisonReport:
    """GBR-vs-Ridge comparison report per category."""

    gbr_metrics: dict[str, float]
    ridge_metrics: dict[str, float]
    winner: str
    rationale: str


def _make_folds(
    unique_weeks: pd.Series,
    n_splits: int,
    train_weeks: int,
    test_weeks: int,
) -> list[tuple[pd.Series, pd.Series, pd.Series]]:
    """Build (train_cut, test_start, test_end) boundary triples for each fold."""
    folds = []
    for split_index in range(n_splits):
        train_end = train_weeks + split_index * test_weeks
        test_end = train_end + test_weeks
        if test_end > len(unique_weeks):
            break
        folds.append((
            unique_weeks.iloc[train_end - 1],
            unique_weeks.iloc[train_end],
            unique_weeks.iloc[test_end - 1],
        ))
    return folds


def _score_fold(
    model_kind: str,
    sorted_df: pd.DataFrame,
    train_cut: pd.Series,
    test_start: pd.Series,
    test_end: pd.Series,
) -> dict[str, float] | None:
    """Train on rows ≤ train_cut, score on rows in [test_start, test_end].

    Returns:
        Metrics dict, or None if slices are empty or features cannot be built.
    """
    train_slice = sorted_df.loc[sorted_df["week"] <= train_cut]
    test_slice = sorted_df.loc[(sorted_df["week"] >= test_start) & (sorted_df["week"] <= test_end)]
    if train_slice.empty or test_slice.empty:
        return None
    bundle = _train_kind(model_kind, train_slice)
    test_features, test_target = build_training_features(test_slice)
    if test_features.empty:
        return None
    aligned = test_features.reindex(columns=bundle.feature_names, fill_value=0.0)
    predictions = bundle.point.predict(aligned)
    return evaluate_predictions(test_target.to_numpy(), predictions)


def walk_forward_validate(
    weekly_df: pd.DataFrame,
    n_splits: int = _DEFAULT_SPLITS,
    train_weeks: int = _DEFAULT_TRAIN_WEEKS,
    test_weeks: int = _DEFAULT_TEST_WEEKS,
    model_kind: str = "gbr",
) -> EvaluationReport:
    """Walk-forward CV: expanding train window, fixed test horizon.

    Raises:
        ValueError: If `model_kind` is unknown or there is insufficient data.
    """
    if model_kind not in {"gbr", "ridge"}:
        raise ValueError(f"unknown model_kind: {model_kind!r}")
    sorted_df = weekly_df.sort_values("week").reset_index(drop=True)
    unique_weeks = sorted_df["week"].drop_duplicates().sort_values().reset_index(drop=True)
    if len(unique_weeks) < train_weeks + test_weeks:
        raise ValueError(
            f"insufficient history: need {train_weeks + test_weeks} weeks, got {len(unique_weeks)}."
        )
    folds: list[FoldResult] = []
    for idx, (train_cut, test_start, test_end) in enumerate(
        _make_folds(unique_weeks, n_splits, train_weeks, test_weeks)
    ):
        metrics = _score_fold(model_kind, sorted_df, train_cut, test_start, test_end)
        if metrics is not None:
            folds.append(FoldResult(fold=idx + 1, train_size=0, test_size=0, metrics=metrics))
    return EvaluationReport(folds=folds, mean_metrics=_aggregate_metrics(folds))


def _train_kind(kind: str, train_slice: pd.DataFrame) -> TrainedForecaster:
    """Train either GBR or Ridge on a temporal slice."""
    features, target = build_training_features(train_slice)
    if kind == "gbr":
        return train_gbr(features, target)
    return train_ridge(features, target)


def _aggregate_metrics(folds: list[FoldResult]) -> dict[str, float]:
    """Average per-fold metrics."""
    if not folds:
        return {"mape": 0.0, "rmse": 0.0, "mae": 0.0, "r2": 0.0}
    keys = folds[0].metrics.keys()
    return {key: float(np.mean([fold.metrics[key] for fold in folds])) for key in keys}


def compare_models(
    weekly_df: pd.DataFrame,
    n_splits: int = _DEFAULT_SPLITS,
    train_weeks: int = _DEFAULT_TRAIN_WEEKS,
    test_weeks: int = _DEFAULT_TEST_WEEKS,
) -> ComparisonReport:
    """Compare GBR vs Ridge with the same walk-forward protocol.

    Args:
        weekly_df: Weekly DataFrame filtered to one category.
        n_splits: Number of walk-forward folds.
        train_weeks: Minimum training window in weeks.
        test_weeks: Test horizon per fold in weeks.

    Returns:
        ComparisonReport. If GBR's MAPE improvement < 5%, the simpler Ridge wins.
    """
    gbr_report = walk_forward_validate(weekly_df, n_splits=n_splits,
                                       train_weeks=train_weeks, test_weeks=test_weeks,
                                       model_kind="gbr")
    ridge_report = walk_forward_validate(weekly_df, n_splits=n_splits,
                                         train_weeks=train_weeks, test_weeks=test_weeks,
                                         model_kind="ridge")
    gbr_mape = gbr_report.mean_metrics.get("mape", 0.0)
    ridge_mape = ridge_report.mean_metrics.get("mape", 0.0)
    improvement_pct = (ridge_mape - gbr_mape) / ridge_mape * 100.0 if ridge_mape > 0 else 0.0
    if improvement_pct < _MAPE_TOLERANCE_PCT:
        winner, rationale = "ridge", (
            f"GBR improved MAPE by only {improvement_pct:.2f}% (< {_MAPE_TOLERANCE_PCT}%). "
            "Simpler Ridge baseline wins."
        )
    else:
        winner, rationale = "gbr", (
            f"GBR improved MAPE by {improvement_pct:.2f}% over Ridge — primary model retained."
        )
    logger.info("model_comparison", extra={"gbr_mape": gbr_mape, "ridge_mape": ridge_mape, "winner": winner})
    return ComparisonReport(gbr_metrics=gbr_report.mean_metrics, ridge_metrics=ridge_report.mean_metrics,
                            winner=winner, rationale=rationale)
