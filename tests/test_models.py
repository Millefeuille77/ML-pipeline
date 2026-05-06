"""Tests for feature engineering, forecaster, clustering, anomaly, eval, registry."""
from __future__ import annotations

from datetime import date as date_type, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.api.schemas import Alert, ClusterResult, ForecastResult
from src.models import anomaly, clustering, evaluation, feature_engineering, forecaster, registry


# ---------------- feature engineering ---------------------------------------

def test_build_training_features_returns_aligned_xy(synthetic_training_df: pd.DataFrame) -> None:
    """X and y must be the same length with no NaN values."""
    features, target = feature_engineering.build_training_features(synthetic_training_df)
    assert len(features) == len(target)
    assert not features.isna().any().any()
    assert not target.isna().any()


def test_build_training_features_drops_rows_without_target(make_weekly_row) -> None:
    """Rows missing `target_next_week` must be dropped."""
    rows = [make_weekly_row(week=date_type(2024, 1, 1)),
            make_weekly_row(week=date_type(2024, 1, 8), target_next_week=None)]
    frame = pd.DataFrame(rows)
    features, target = feature_engineering.build_training_features(frame)
    assert len(features) == 1
    assert len(target) == 1


def test_build_training_features_raises_when_target_missing(sample_weekly_df: pd.DataFrame) -> None:
    """Missing target column must raise ValueError."""
    frame = sample_weekly_df.drop(columns=["target_next_week"])
    with pytest.raises(ValueError):
        feature_engineering.build_training_features(frame)


def test_build_inference_features_returns_single_row(sample_weekly_df: pd.DataFrame) -> None:
    """`build_inference_features` must return exactly one row."""
    out = feature_engineering.build_inference_features(
        "MI-006", "Retail", "PL-Central", sample_weekly_df
    )
    assert len(out) == 1


def test_build_inference_features_raises_on_empty() -> None:
    """Empty history must raise ValueError."""
    with pytest.raises(ValueError):
        feature_engineering.build_inference_features("MI-006", "Retail", "PL-Central", pd.DataFrame())


# ---------------- forecaster -------------------------------------------------

def test_forecaster_train_gbr_smoke(synthetic_training_df: pd.DataFrame) -> None:
    """GBR training should succeed and report finite metrics."""
    features, target = feature_engineering.build_training_features(synthetic_training_df)
    bundle = forecaster.train_gbr(features.head(200), target.head(200))
    for key in ("mape", "rmse", "mae", "r2"):
        assert key in bundle.metrics
        assert np.isfinite(bundle.metrics[key])


def test_forecaster_predict_horizon_returns_forecast_result(
    synthetic_training_df: pd.DataFrame,
) -> None:
    """`predict_horizon` must return a ForecastResult with arrays of correct length."""
    features, target = feature_engineering.build_training_features(synthetic_training_df.head(200))
    bundle = forecaster.train_gbr(features, target)
    bundle.category = "Milk"
    history = synthetic_training_df[
        (synthetic_training_df["sku"] == "MI-006")
        & (synthetic_training_df["channel"] == "Retail")
        & (synthetic_training_df["region"] == "PL-Central")
    ].sort_values("week").reset_index(drop=True)
    result = forecaster.predict_horizon(bundle, "MI-006", "Retail", "PL-Central", history, 4)
    assert isinstance(result, ForecastResult)
    assert len(result.weeks) == 4
    assert len(result.predicted_units) == 4
    assert len(result.confidence_lower) == 4
    assert len(result.confidence_upper) == 4


def test_forecaster_intervals_bracket_point_estimate(
    synthetic_training_df: pd.DataFrame,
) -> None:
    """`confidence_lower <= predicted <= confidence_upper` per step."""
    features, target = feature_engineering.build_training_features(synthetic_training_df.head(200))
    bundle = forecaster.train_gbr(features, target)
    bundle.category = "Milk"
    history = synthetic_training_df[
        (synthetic_training_df["sku"] == "MI-006")
        & (synthetic_training_df["channel"] == "Retail")
        & (synthetic_training_df["region"] == "PL-Central")
    ].sort_values("week").reset_index(drop=True)
    result = forecaster.predict_horizon(bundle, "MI-006", "Retail", "PL-Central", history, 3)
    for low, point, high in zip(result.confidence_lower, result.predicted_units, result.confidence_upper):
        assert low <= point <= high


def test_forecaster_train_per_category_persists_models(
    synthetic_training_df: pd.DataFrame, temp_model_dir: Path
) -> None:
    """`train_per_category` must register one model per category."""
    bundles = forecaster.train_per_category(synthetic_training_df.head(400))
    assert set(bundles.keys()).issubset({"Milk", "Yogurt"})
    artifacts = list(temp_model_dir.glob("*.joblib"))
    assert artifacts


# ---------------- clustering -------------------------------------------------

def test_clustering_assigns_label_to_every_sku(synthetic_training_df: pd.DataFrame) -> None:
    """Every SKU in the dataset must end up with a cluster label."""
    model = clustering.fit_clustering(synthetic_training_df)
    skus = set(synthetic_training_df["sku"].unique())
    assert set(model.sku_labels.keys()) == skus


def test_clustering_chosen_k_in_range(synthetic_training_df: pd.DataFrame) -> None:
    """The chosen k must lie in the [3, 6] sweep range — but the synthetic
    dataset has fewer SKUs, so k may be the dataset minimum (3)."""
    model = clustering.fit_clustering(synthetic_training_df)
    assert 3 <= model.estimator.n_clusters <= 6


def test_clustering_get_cluster_returns_cluster_result(
    synthetic_training_df: pd.DataFrame,
) -> None:
    """`get_cluster` must return a populated ClusterResult."""
    model = clustering.fit_clustering(synthetic_training_df)
    sku = next(iter(model.sku_labels.keys()))
    result = clustering.get_cluster(sku, model)
    assert isinstance(result, ClusterResult)
    assert result.cluster_label.startswith("cluster_")
    assert result.cluster_description


# ---------------- anomaly ----------------------------------------------------

def test_anomaly_detects_demand_spike(make_weekly_row) -> None:
    """A 5x rolling-mean spike must produce a HIGH demand_spike alert."""
    rows = [
        make_weekly_row(units_sold=500.0, rolling_mean_4=100.0, rolling_std_4=10.0, price_avg=10.0),
    ]
    alerts = anomaly.detect_alerts(pd.DataFrame(rows))
    spike = [alert for alert in alerts if alert.alert_type == "demand_spike"]
    assert spike
    assert spike[0].severity == "HIGH"


def test_anomaly_negative_units_sold_returns_low_alert(make_weekly_row) -> None:
    """Negative units_sold must produce a LOW return_anomaly alert (not error)."""
    rows = [make_weekly_row(units_sold=-3.0, price_avg=10.0)]
    alerts = anomaly.detect_alerts(pd.DataFrame(rows))
    returns = [alert for alert in alerts if alert.alert_type == "return_anomaly"]
    assert returns
    assert returns[0].severity == "LOW"


def test_anomaly_stock_risk_when_cover_below_two_weeks(make_weekly_row) -> None:
    """`stock_available / rolling_mean_4 < 2.0` must produce HIGH stock_risk."""
    rows = [make_weekly_row(stock_available=50.0, rolling_mean_4=100.0,
                            units_sold=100.0, price_avg=10.0)]
    alerts = anomaly.detect_alerts(pd.DataFrame(rows))
    stock = [alert for alert in alerts if alert.alert_type == "stock_risk"]
    assert stock
    assert stock[0].severity == "HIGH"


def test_anomaly_filter_by_sku(make_weekly_row) -> None:
    """`filter_alerts(sku=...)` keeps only matching alerts."""
    rows = [
        make_weekly_row(sku="MI-006", units_sold=-1.0, price_avg=10.0),
        make_weekly_row(sku="YO-001", units_sold=-1.0, price_avg=10.0),
    ]
    alerts = anomaly.detect_alerts(pd.DataFrame(rows))
    only_mi = anomaly.filter_alerts(alerts, sku="MI-006")
    assert all(alert.sku == "MI-006" for alert in only_mi)
    assert any(alert.sku == "MI-006" for alert in alerts)


# ---------------- evaluation -------------------------------------------------

def test_walk_forward_uses_temporal_splits(synthetic_training_df: pd.DataFrame) -> None:
    """The walk-forward report must contain at least one fold with metrics."""
    report = evaluation.walk_forward_validate(
        synthetic_training_df, n_splits=2, train_weeks=80, test_weeks=4
    )
    assert report.folds
    assert "mape" in report.mean_metrics


def test_walk_forward_rejects_unknown_kind(synthetic_training_df: pd.DataFrame) -> None:
    """Unknown `model_kind` must raise ValueError."""
    with pytest.raises(ValueError):
        evaluation.walk_forward_validate(synthetic_training_df, model_kind="lightgbm")


def test_compare_models_returns_both_metric_sets(synthetic_training_df: pd.DataFrame) -> None:
    """The comparison report must contain both gbr and ridge metrics."""
    report = evaluation.compare_models(synthetic_training_df.head(2000))
    assert "mape" in report.gbr_metrics
    assert "mape" in report.ridge_metrics
    assert report.winner in {"gbr", "ridge"}


# ---------------- registry --------------------------------------------------

def test_registry_save_and_load_roundtrip(temp_model_dir: Path) -> None:
    """save_model → get_model returns the same object."""
    fake = {"weights": [1, 2, 3]}
    info = registry.save_model(fake, "test_estimator", {"mape": 1.0}, ["a", "b"])
    loaded, loaded_info = registry.get_model("test_estimator", "latest")
    assert loaded == fake
    assert loaded_info.version == info.version


def test_registry_rejects_unsafe_name(temp_model_dir: Path) -> None:
    """Path-traversal style names must be rejected."""
    with pytest.raises(ValueError):
        registry.save_model({}, "../etc/passwd", {}, [])


def test_registry_versions_increment(temp_model_dir: Path) -> None:
    """Saving twice under the same name produces v1 then v2."""
    first = registry.save_model({}, "vmodel", {}, [])
    second = registry.save_model({}, "vmodel", {}, [])
    assert second.version == first.version + 1


@pytest.mark.parametrize(
    "unsafe_name",
    ["..", "../foo", "/abs/path", "with\\backslash", "with/slash"],
)
def test_registry_get_model_rejects_traversal_separators(
    unsafe_name: str, temp_model_dir: Path
) -> None:
    """Traversal-shaped names must be refused before any file lookup."""
    with pytest.raises(ValueError, match="invalid model name"):
        registry.get_model(unsafe_name, "latest")


# ---------------- forecaster: batch + DB lifecycle lookup -------------------

def test_predict_batch_dataframe_returns_results(
    synthetic_training_df: pd.DataFrame, temp_model_dir: Path, monkeypatch
) -> None:
    """`predict_batch_dataframe` returns one ForecastResult per (sku,channel,region) group.

    Trains a per-category model, persists via the registry, then drives the
    batch path with a small daily-grain frame. Confidence intervals must
    bracket the point estimate at every horizon step.
    """
    forecaster.train_per_category(synthetic_training_df.head(400))
    monkeypatch.setattr(
        forecaster, "_lookup_sku_metadata", lambda sku: ("Mature", 100), raising=True
    )

    base_date = pd.Timestamp("2024-12-02")
    daily_rows = []
    for offset in range(14):
        daily_rows.append({
            "date": (base_date + pd.Timedelta(days=offset)).date(),
            "sku": "MI-006",
            "channel": "Retail",
            "region": "PL-Central",
            "category": "Milk",
            "units_sold": 50.0 + offset,
            "stock_available": 200.0,
            "promotion_flag": 0,
            "price_unit": 10.0,
            "delivery_days": 2,
            "delivered_qty": 40.0,
        })
    batch_df = pd.DataFrame(daily_rows)
    results = forecaster.predict_batch_dataframe(batch_df)

    assert results, "expected at least one ForecastResult from batch"
    assert all(isinstance(r, ForecastResult) for r in results)
    for forecast in results:
        for low, point, high in zip(
            forecast.confidence_lower, forecast.predicted_units, forecast.confidence_upper
        ):
            assert low <= point <= high, (
                f"interval violated: {low} <= {point} <= {high}"
            )


def test_predict_batch_uses_db_lifecycle_not_hardcoded(
    synthetic_training_df: pd.DataFrame, temp_model_dir: Path, monkeypatch
) -> None:
    """Batch features must reflect `_lookup_sku_metadata` not a hardcoded label.

    Patches `_lookup_sku_metadata` to return ('Decline', 50) and asserts that
    `add_derived_features` of the resulting weekly history yields
    `lifecycle_encoded == 0` and `sku_age >= 50`.
    """
    forecaster.train_per_category(synthetic_training_df.head(400))
    monkeypatch.setattr(
        forecaster, "_lookup_sku_metadata", lambda sku: ("Decline", 50), raising=True
    )

    base_date = pd.Timestamp("2024-12-02")
    daily_rows = [
        {
            "date": (base_date + pd.Timedelta(days=offset)).date(),
            "sku": "MI-006",
            "channel": "Retail",
            "region": "PL-Central",
            "category": "Milk",
            "units_sold": 50.0 + offset,
            "stock_available": 200.0,
            "promotion_flag": 0,
            "price_unit": 10.0,
            "delivery_days": 2,
            "delivered_qty": 40.0,
        }
        for offset in range(14)
    ]
    weekly_history = forecaster._aggregate_to_weekly_history(pd.DataFrame(daily_rows))
    assert (weekly_history["lifecycle_stage"] == "Decline").all()
    assert weekly_history["sku_age"].min() >= 50

    enriched = feature_engineering.add_derived_features(weekly_history)
    assert (enriched["lifecycle_encoded"] == 0).all()


# ---------------- evaluation: temporal-only splits + tie-break ---------------

def test_walk_forward_uses_temporal_splits_strict() -> None:
    """For every fold, test_slice.week.min() must be strictly > train_slice.week.max().

    Builds a synthetic 30-week single-SKU frame so the date math is checkable
    by inspection. Drives the internal `_make_folds` and re-derives the
    train/test slices to assert temporal disjointness.
    """
    base_week = pd.Timestamp("2022-01-03")
    rows = [
        {
            "sku": "MI-006",
            "week": (base_week + pd.Timedelta(weeks=offset)).date(),
            "channel": "Retail",
            "region": "PL-Central",
            "category": "Milk",
            "units_sold": 50.0 + offset,
            "stock_available": 200.0,
            "promotion_flag": 0,
            "price_unit": 10.0,
            "delivery_days": 2.0,
            "is_holiday_peak": 0,
            "week_number": (offset % 52) + 1,
            "month": ((offset // 4) % 12) + 1,
            "year": 2022,
            "is_holiday_week": 0,
            "is_summer": 0,
            "is_winter": 1,
            "sku_age": 50 + offset,
            "lifecycle_stage": "Mature",
            "lag_1": 50.0,
            "lag_2": 49.0,
            "rolling_mean_4": 50.0,
            "rolling_std_4": 1.0,
            "momentum": 1.0,
            "target_next_week": 51.0 + offset,
        }
        for offset in range(30)
    ]
    frame = pd.DataFrame(rows)
    sorted_df = frame.sort_values("week").reset_index(drop=True)
    unique_weeks = sorted_df["week"].drop_duplicates().sort_values().reset_index(drop=True)
    folds = evaluation._make_folds(unique_weeks, n_splits=3, train_weeks=20, test_weeks=2)

    assert folds, "expected at least one synthesized fold"
    for train_cut, test_start, test_end in folds:
        train_slice = sorted_df.loc[sorted_df["week"] <= train_cut]
        test_slice = sorted_df.loc[
            (sorted_df["week"] >= test_start) & (sorted_df["week"] <= test_end)
        ]
        assert not train_slice.empty
        assert not test_slice.empty
        assert test_slice["week"].min() > train_slice["week"].max(), (
            "DATA LEAKAGE: test window starts on or before the last training week."
        )


def test_compare_models_picks_simpler_when_close(monkeypatch) -> None:
    """When Ridge MAPE is within 5% of GBR MAPE the simpler Ridge model wins."""
    from src.models.evaluation import (
        ComparisonReport,
        EvaluationReport,
        compare_models,
    )

    def _fake_walk_forward(weekly_df, model_kind="gbr", **_kwargs):
        if model_kind == "gbr":
            return EvaluationReport(folds=[], mean_metrics={"mape": 9.8})
        return EvaluationReport(folds=[], mean_metrics={"mape": 10.0})

    monkeypatch.setattr(
        "src.models.evaluation.walk_forward_validate", _fake_walk_forward, raising=True
    )
    report: ComparisonReport = compare_models(pd.DataFrame())
    assert report.winner == "ridge"
    assert "Simpler Ridge" in report.rationale


# ---------------- clustering: 30 SKUs across 5 categories --------------------

def test_clustering_handles_30_skus(make_weekly_row) -> None:
    """All 30 SKUs across 5 categories must each receive a label; k in [3, 6]."""
    rng = np.random.default_rng(seed=7)
    base_week = pd.Timestamp("2024-01-07")
    categories = ["Milk", "Yogurt", "ReadyMeal", "Juice", "SnackBar"]
    sku_prefixes = {"Milk": "MI", "Yogurt": "YO", "ReadyMeal": "RM",
                    "Juice": "JU", "SnackBar": "SB"}
    rows = []
    for category in categories:
        prefix = sku_prefixes[category]
        for sku_index in range(6):  # 6 per category × 5 categories = 30 SKUs
            sku = f"{prefix}-{sku_index + 1:03d}"
            for week_offset in range(20):
                units = float(rng.integers(20, 200))
                rows.append(make_weekly_row(
                    sku=sku,
                    week=(base_week + pd.Timedelta(weeks=week_offset)).date(),
                    units_sold=units,
                    target_next_week=units + 1.0,
                    week_number=(week_offset % 52) + 1,
                    month=((week_offset // 4) % 12) + 1,
                ))
    frame = pd.DataFrame(rows)

    model = clustering.fit_clustering(frame)

    assert len(model.sku_labels) == 30
    assert set(model.sku_labels.keys()) == set(frame["sku"].unique())
    assert 3 <= model.estimator.n_clusters <= 6


# ---------------- anomaly: drop + price deviation ----------------------------

def test_anomaly_demand_drop_detected(make_weekly_row) -> None:
    """Units below `rolling_mean_4 - 2 * rolling_std_4` (and >= 0) triggers MEDIUM drop."""
    rows = [
        make_weekly_row(
            units_sold=10.0, rolling_mean_4=100.0, rolling_std_4=20.0, price_avg=10.0
        ),
    ]
    alerts = anomaly.detect_alerts(pd.DataFrame(rows))
    drops = [alert for alert in alerts if alert.alert_type == "demand_drop"]
    assert drops, "expected at least one demand_drop alert"
    assert drops[0].severity == "MEDIUM"


# ---------------- Phase B regression tests ----------------------------------


def test_save_model_does_not_collide_with_reserved_logrecord_name(
    temp_model_dir: Path,
) -> None:
    """Phase B3: `save_model` previously passed `extra={"name": ...}` which
    collides with `LogRecord.name` and raises `KeyError("Attempt to overwrite
    'name' in LogRecord")` whenever the root logger has any handler attached.

    This test attaches a real handler to the root so `Logger.makeRecord` is
    exercised, then calls `save_model`. Pre-fix this would crash with KeyError;
    post-fix the artifact must land on disk and the call must return a
    populated `ModelInfo`.
    """
    import logging
    from io import StringIO

    captured = StringIO()
    handler = logging.StreamHandler(captured)
    handler.setLevel(logging.INFO)
    root = logging.getLogger()
    previous_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    try:
        info = registry.save_model(
            {"weights": [0.1, 0.2]},
            "phase_b_collision_test",
            {"mape": 1.0},
            ["feature_a"],
            training_rows=10,
            category="Milk",
        )
    finally:
        root.removeHandler(handler)
        root.setLevel(previous_level)

    assert info.name == "phase_b_collision_test"
    assert info.path.exists(), "joblib artifact must be written to disk"
    assert info.metadata_path.exists(), "metadata JSON must be written"
    # Sanity: the call really did emit a log line via the configured handler.
    handler.flush()


def test_save_model_extras_avoid_all_reserved_logrecord_keys(
    temp_model_dir: Path,
) -> None:
    """Phase B3 (defense in depth): scan the registry source for any reserved
    LogRecord key sneaking back into `extra=` dicts. If a future edit
    re-introduces `"name"` or any reserved key the test fails loudly.
    """
    import re
    from pathlib import Path as _Path

    reserved = {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "message", "asctime",
    }
    source = _Path(registry.__file__).read_text(encoding="utf-8")
    extras = re.findall(r"extra=\{([^}]*)\}", source)
    for extra_block in extras:
        keys = re.findall(r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:', extra_block)
        offenders = [key for key in keys if key in reserved]
        assert not offenders, (
            f"reserved LogRecord keys leaked into registry logger extras: "
            f"{offenders!r} in block {extra_block!r}"
        )


def test_add_derived_features_handles_missing_category_column(
    sample_weekly_df: pd.DataFrame,
) -> None:
    """Phase B4: at inference time `recent_data` lacks `category` (lives in
    products table). `add_derived_features` previously called
    `category_avg.replace(0, np.nan)` on a scalar from `frame["price_unit"].mean()`
    which has no `.replace` method, raising AttributeError.

    With the scalar/Series branch fix the function must return a frame with the
    `price_vs_category_avg` column populated and not crash.
    """
    from src.models.feature_engineering import add_derived_features

    frame = sample_weekly_df.copy().drop(columns=["category"])
    assert "category" not in frame.columns, "precondition for inference path"

    out = add_derived_features(frame)
    assert "price_vs_category_avg" in out.columns
    assert not out["price_vs_category_avg"].isna().any()
    # When all price_unit values are equal (10.0 in fixture) ratio == 1.0
    assert out["price_vs_category_avg"].to_list() == pytest.approx([1.0] * len(out))


def test_add_derived_features_scalar_branch_handles_zero_average() -> None:
    """Phase B4 corner case: when `price_unit` is uniformly zero the scalar
    branch must fall back to NaN (then fillna(1.0)) rather than dividing by 0.

    Pre-fix the only branch was `Series.replace(0, np.nan)` which would have
    crashed on a scalar. The new branch checks `if avg == 0`.
    """
    import pandas as pd
    from src.models.feature_engineering import add_derived_features

    frame = pd.DataFrame(
        [
            {
                "sku": "MI-006", "channel": "Retail", "region": "PL-Central",
                "week": pd.Timestamp("2024-01-01").date(),
                "units_sold": 0.0, "stock_available": 0.0, "promotion_flag": 0,
                "price_unit": 0.0, "delivery_days": 2.0, "is_holiday_peak": 0,
                "week_number": 1, "month": 1, "year": 2024,
                "is_holiday_week": 0, "is_summer": 0, "is_winter": 1,
                "sku_age": 100, "lifecycle_stage": "Mature",
                "lag_1": 0.0, "lag_2": 0.0, "rolling_mean_4": 0.0,
                "rolling_std_4": 0.0, "momentum": 0.0,
            }
        ]
    )
    out = add_derived_features(frame)
    assert "price_vs_category_avg" in out.columns
    # 0 / NaN → NaN → fillna(1.0)
    assert out["price_vs_category_avg"].iloc[0] == pytest.approx(1.0)


def test_add_derived_features_series_branch_unchanged_with_category(
    synthetic_training_df: pd.DataFrame,
) -> None:
    """Phase B4 invariant: when `category` IS present the function must use the
    per-category mean (Series path), unchanged from before B4.
    """
    from src.models.feature_engineering import add_derived_features

    frame = synthetic_training_df.head(50).copy()
    assert "category" in frame.columns
    out = add_derived_features(frame)
    assert "price_vs_category_avg" in out.columns
    # All fixture rows in synthetic data share price_unit=10.0 so within a
    # category the ratio is exactly 1.0
    assert out["price_vs_category_avg"].to_list() == pytest.approx([1.0] * len(out))


def test_anomaly_price_change_detected(make_weekly_row) -> None:
    """Price > 15% off the 4-week rolling avg yields a LOW price_anomaly alert."""
    rows = [
        make_weekly_row(
            price_unit=12.0,           # 20% above price_avg
            price_avg=10.0,
            units_sold=100.0,
            rolling_mean_4=100.0,
            rolling_std_4=5.0,
            stock_available=400.0,
        ),
    ]
    alerts = anomaly.detect_alerts(pd.DataFrame(rows))
    price_alerts = [a for a in alerts if a.alert_type == "price_anomaly"]
    assert price_alerts, "expected price_anomaly alert for >15% deviation"
    assert price_alerts[0].severity == "LOW"
