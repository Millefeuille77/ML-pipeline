"""Phase D Level-1 MLOps script unit tests.

Targets:
- scripts.synthesize_weekly_data : data generator (preserves negatives, mean fidelity)
- scripts.validate_data          : pre-training validation gate
- scripts.score_predictions      : daily scoring + metric upsert

All tests run on a bare CI Ubuntu host with no DB and no container — the
session_scope / engine surface is patched at module level.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytest


# =================== shared fakes ==========================================


class _RecordingSession:
    """Minimal session that records every (sql_text, payload) pair without raising."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def execute(self, statement, payload=None):
        sql = str(getattr(statement, "text", statement))
        self.calls.append((sql, payload))
        return self

    def fetchall(self) -> list:
        return []

    def scalar(self) -> Any:
        return None

    def commit(self) -> None:  # pragma: no cover
        pass

    def rollback(self) -> None:  # pragma: no cover
        pass

    def close(self) -> None:  # pragma: no cover
        pass


# =================== synthesize_weekly_data tests ===========================


def _stub_synth_db(monkeypatch, *, stats_df: pd.DataFrame,
                   lifecycle_df: pd.DataFrame, seasonality_df: pd.DataFrame,
                   latest_week: date) -> None:
    """Stub the three SQL helpers in synthesize_weekly_data so the generator
    runs without a live DB."""
    from scripts import synthesize_weekly_data as syn

    monkeypatch.setattr(syn, "_load_series_stats", lambda _s: stats_df, raising=True)
    monkeypatch.setattr(syn, "_load_lifecycle", lambda _s: lifecycle_df, raising=True)
    monkeypatch.setattr(syn, "_load_seasonality", lambda _s: seasonality_df, raising=True)
    monkeypatch.setattr(syn, "_latest_db_week", lambda _s: latest_week, raising=True)

    @contextmanager
    def _scope():
        yield _RecordingSession()

    monkeypatch.setattr(syn, "session_scope", _scope, raising=True)


@pytest.fixture
def synth_inputs() -> dict[str, pd.DataFrame]:
    """Build the three stat frames the synthesizer consumes for one SKU series."""
    stats = pd.DataFrame([{
        "sku": "MI-006", "channel": "Retail", "region": "PL-Central",
        "mean_us": 100.0, "std_us": 20.0, "min_us": -5.0, "max_us": 200.0,
        "mean_price": 10.0, "mean_stock": 500.0, "mean_dd": 2.0,
        "max_sku_age": 100, "latest_week": date(2024, 12, 23),
    }])
    lifecycle = pd.DataFrame([{"sku": "MI-006", "lifecycle_stage": "Mature"}])
    seasonality_rows = [
        {"sku": "MI-006", "channel": "Retail", "region": "PL-Central",
         "month": m, "seasonality_factor": 1.0}
        for m in range(1, 13)
    ]
    seasonality = pd.DataFrame(seasonality_rows)
    return {"stats": stats, "lifecycle": lifecycle, "seasonality": seasonality}


def test_synthesize_preserves_sku_mean_within_one_std(monkeypatch, synth_inputs) -> None:
    """Generated `units_sold` mean must lie within ±1 historical std of the
    historical mean for each SKU series. With seasonality_factor==1.0 across
    the year the generator draws from N(mean, std*0.5) clipped to [min_us*0.5,
    max_us*1.5], so the empirical mean must be within ±1σ of mean_us=100.
    """
    from scripts.synthesize_weekly_data import synthesize

    _stub_synth_db(
        monkeypatch,
        stats_df=synth_inputs["stats"],
        lifecycle_df=synth_inputs["lifecycle"],
        seasonality_df=synth_inputs["seasonality"],
        latest_week=date(2024, 12, 23),
    )
    df = synthesize(n_weeks=100, seed=42, insert=False)
    assert len(df) == 100  # one series × 100 weeks
    historical_mean = 100.0
    historical_std = 20.0
    empirical_mean = float(df["units_sold"].mean())
    assert abs(empirical_mean - historical_mean) <= historical_std, (
        f"synth mean {empirical_mean:.2f} drifts > 1 std from historical "
        f"{historical_mean:.2f} (std={historical_std:.2f})"
    )


def test_synthesize_can_produce_negative_units_sold(monkeypatch, synth_inputs) -> None:
    """Returns are valid: the generator must NOT clamp `units_sold` to >= 0.

    With `min_us = -5.0` and clip_low = 0.5, the lower bound is -2.5, so over
    a sufficiently large draw at least one negative value should appear when
    the mean is shifted close to the negative side.
    """
    from scripts.synthesize_weekly_data import synthesize

    # Force the distribution so negatives are likely: mean=0, large std, neg min.
    stats = pd.DataFrame([{
        "sku": "MI-006", "channel": "Retail", "region": "PL-Central",
        "mean_us": 0.0, "std_us": 50.0, "min_us": -100.0, "max_us": 100.0,
        "mean_price": 10.0, "mean_stock": 500.0, "mean_dd": 2.0,
        "max_sku_age": 100, "latest_week": date(2024, 12, 23),
    }])
    _stub_synth_db(
        monkeypatch,
        stats_df=stats,
        lifecycle_df=synth_inputs["lifecycle"],
        seasonality_df=synth_inputs["seasonality"],
        latest_week=date(2024, 12, 23),
    )
    df = synthesize(n_weeks=200, seed=42, insert=False)
    negatives = df[df["units_sold"] < 0]
    assert len(negatives) >= 1, (
        f"generator should preserve negative units_sold (returns) but produced "
        f"min={df['units_sold'].min():.2f}; clamping to 0 would be a regression"
    )


def test_synthesize_starts_after_latest_week(monkeypatch, synth_inputs) -> None:
    """Generator must produce dates strictly AFTER the latest DB week
    (idempotent: never overwrites existing data)."""
    from scripts.synthesize_weekly_data import synthesize

    latest = date(2024, 12, 23)
    _stub_synth_db(
        monkeypatch,
        stats_df=synth_inputs["stats"],
        lifecycle_df=synth_inputs["lifecycle"],
        seasonality_df=synth_inputs["seasonality"],
        latest_week=latest,
    )
    df = synthesize(n_weeks=4, seed=1, insert=False)
    assert (df["week"] > latest).all(), (
        f"all synthetic weeks must be after {latest}, got min={df['week'].min()}"
    )
    # First week must equal latest + 7 days.
    assert df["week"].min() == latest + timedelta(weeks=1)


# =================== validate_data tests ===================================


def test_validate_data_fails_on_missing_lifecycle_column(monkeypatch) -> None:
    """ValidationReport.passed must be False when `lifecycle_stage` is missing."""
    from scripts import validate_data as vd

    minimal_rows = [{
        "sku": "MI-006", "week": date(2024, 12, 23), "channel": "Retail",
        "region": "PL-Central", "units_sold": 100.0, "stock_available": 500.0,
        "promotion_flag": 0, "price_unit": 10.0, "delivery_days": 2.0,
        "is_holiday_peak": 0, "week_number": 51, "month": 12, "year": 2024,
        "is_holiday_week": 0, "is_summer": 0, "is_winter": 1,
        "sku_age": 100,
        # lifecycle_stage intentionally absent
        "lag_1": 95.0, "lag_2": 90.0, "rolling_mean_4": 95.0,
        "rolling_std_4": 5.0, "momentum": 5.0, "target_next_week": 105.0,
    }]
    df = pd.DataFrame(minimal_rows)
    monkeypatch.setattr(vd, "_load_weekly_features", lambda: df, raising=True)

    report = vd.validate(allow_stale=True)
    assert report.passed is False, "missing lifecycle_stage must FAIL the report"
    failed = [c for c in report.checks if not c["ok"] and c["level"] == "FAIL"]
    schema_fail = [c for c in failed if c["name"] == "schema_columns"]
    assert schema_fail, (
        f"expected a 'schema_columns' FAIL check naming the missing column, got: "
        f"{[c['name'] for c in failed]}"
    )
    assert "lifecycle_stage" in schema_fail[0]["message"]


def test_validate_data_passes_on_complete_frame(monkeypatch, synthetic_training_df) -> None:
    """A complete fixture frame matching the schema must pass validation
    (cardinality WARNs from the synthetic-fixture limited SKU set are allowed)."""
    from scripts import validate_data as vd

    df = synthetic_training_df.copy()
    monkeypatch.setattr(vd, "_load_weekly_features", lambda: df, raising=True)
    report = vd.validate(allow_stale=True)
    fails = [c for c in report.checks if not c["ok"] and c["level"] == "FAIL"]
    assert not fails, (
        f"complete frame must not produce FAILs, got: {[c['name'] for c in fails]}"
    )
    assert report.passed is True


def test_validate_data_warns_only_on_distribution_drift(monkeypatch, synthetic_training_df) -> None:
    """Distribution checks must be WARN-level (never block the pipeline)."""
    from scripts import validate_data as vd

    df = synthetic_training_df.copy()
    # Force distribution out of bounds: units_sold mean below the 50.0 floor.
    df["units_sold"] = 1.0
    monkeypatch.setattr(vd, "_load_weekly_features", lambda: df, raising=True)
    report = vd.validate(allow_stale=True)
    dist_check = next(c for c in report.checks if c["name"] == "dist_units_sold")
    assert dist_check["level"] == "WARN", (
        "distribution drift must be WARN, never FAIL — pipeline should still run"
    )
    assert dist_check["ok"] is False
    # Even with a WARN, report.passed stays True (only FAIL flips it).
    fails = [c for c in report.checks if not c["ok"] and c["level"] == "FAIL"]
    if not fails:
        assert report.passed is True


# =================== score_predictions tests ================================


def test_score_predictions_compute_metrics_handles_zero_actuals() -> None:
    """When actual_units == 0 for every row, MAPE must NOT raise ZeroDivisionError —
    the implementation falls back to 0.0 (documented in code)."""
    from scripts.score_predictions import _compute_metrics

    joined = pd.DataFrame([
        {"model_version": "forecaster_snackbar_v1", "category": "SnackBar",
         "predicted_units": 50.0, "actual_units": 0.0},
        {"model_version": "forecaster_snackbar_v1", "category": "SnackBar",
         "predicted_units": 30.0, "actual_units": 0.0},
    ])
    metrics = _compute_metrics(joined)
    assert len(metrics) == 1
    row = metrics.iloc[0]
    assert row["live_mape"] == 0.0, (
        f"all-zero actuals must short-circuit MAPE to 0.0 fallback, got {row['live_mape']}"
    )
    assert row["samples"] == 2
    # MAE/RMSE are still well-defined: |0 - 50| and |0 - 30|.
    assert row["live_mae"] == pytest.approx(40.0)


def test_score_predictions_compute_metrics_correct_mape_with_mixed_actuals() -> None:
    """MAPE skips zero-actual rows but uses non-zero rows correctly.

    Inputs: predicted=[100,50,80], actual=[100,0,100] (one zero).
    Expected MAPE only over rows 0 and 2: ((|0|/100) + (|20|/100)) / 2 = 0.10.
    Expected MAE (all rows): (0 + 50 + 20) / 3 = 23.333...
    Expected RMSE (all rows): sqrt((0 + 2500 + 400)/3) = sqrt(966.67) ≈ 31.09
    """
    from scripts.score_predictions import _compute_metrics

    joined = pd.DataFrame([
        {"model_version": "v1", "category": "Milk",
         "predicted_units": 100.0, "actual_units": 100.0},
        {"model_version": "v1", "category": "Milk",
         "predicted_units": 50.0, "actual_units": 0.0},
        {"model_version": "v1", "category": "Milk",
         "predicted_units": 80.0, "actual_units": 100.0},
    ])
    metrics = _compute_metrics(joined)
    row = metrics.iloc[0]
    assert row["samples"] == 3
    assert row["live_mape"] == pytest.approx(0.10, rel=1e-6)
    assert row["live_mae"] == pytest.approx((0 + 50 + 20) / 3.0)
    assert row["live_rmse"] == pytest.approx(((0 + 2500 + 400) / 3.0) ** 0.5, rel=1e-6)


def test_score_predictions_dry_run_skips_writes(monkeypatch) -> None:
    """`run_scoring(dry_run=True)` must call `_compute_metrics` but never
    open a write transaction. We assert by patching session_scope to raise
    if entered — the dry-run branch must avoid it entirely."""
    from scripts import score_predictions as sp

    # Three unscored predictions referencing two distinct SKUs.
    unscored = pd.DataFrame([
        {"id": 1, "sku": "MI-006", "channel": "Retail", "region": "PL-Central",
         "forecast_week": date(2024, 1, 1),
         "predicted_units": 100.0, "model_version": "v1", "category": "Milk"},
        {"id": 2, "sku": "MI-006", "channel": "Retail", "region": "PL-Central",
         "forecast_week": date(2024, 1, 8),
         "predicted_units": 110.0, "model_version": "v1", "category": "Milk"},
    ])
    actuals = pd.DataFrame([
        {"sku": "MI-006", "channel": "Retail", "region": "PL-Central",
         "week_start": date(2024, 1, 1), "actual_units": 95.0},
        {"sku": "MI-006", "channel": "Retail", "region": "PL-Central",
         "week_start": date(2024, 1, 8), "actual_units": 105.0},
    ])

    class _FakeEngine:
        def connect(self):  # pragma: no cover - not invoked by dry-run path
            raise AssertionError("engine.connect() must not be called in dry-run path")

    monkeypatch.setattr(sp, "_fetch_unscored",
                        lambda _engine: unscored.copy(), raising=True)
    monkeypatch.setattr(sp, "_fetch_actuals",
                        lambda _engine, _skus, _weeks: actuals.copy(), raising=True)
    monkeypatch.setattr(sp, "get_engine", lambda: _FakeEngine(), raising=True)

    @contextmanager
    def _exploding_scope():
        raise AssertionError("session_scope must not be entered in dry-run mode")
        yield  # pragma: no cover

    monkeypatch.setattr(sp, "session_scope", _exploding_scope, raising=True)

    # Must NOT raise — dry-run path skips the session_scope context.
    sp.run_scoring(dry_run=True)


def test_score_predictions_full_run_updates_and_upserts(monkeypatch) -> None:
    """A non-dry-run pass must:
    1. Update prediction_log rows with `actual_units` + `scored_at = NOW()`.
    2. UPSERT a model_performance_live row with computed metrics.
    Both writes use parameterized SQL — we verify by capturing payloads.
    """
    from scripts import score_predictions as sp

    unscored = pd.DataFrame([
        {"id": 1, "sku": "MI-006", "channel": "Retail", "region": "PL-Central",
         "forecast_week": date(2024, 1, 1),
         "predicted_units": 100.0, "model_version": "forecaster_milk_v1",
         "category": "Milk"},
        {"id": 2, "sku": "MI-006", "channel": "Retail", "region": "PL-Central",
         "forecast_week": date(2024, 1, 8),
         "predicted_units": 110.0, "model_version": "forecaster_milk_v1",
         "category": "Milk"},
    ])
    actuals = pd.DataFrame([
        {"sku": "MI-006", "channel": "Retail", "region": "PL-Central",
         "week_start": date(2024, 1, 1), "actual_units": 100.0},
        {"sku": "MI-006", "channel": "Retail", "region": "PL-Central",
         "week_start": date(2024, 1, 8), "actual_units": 90.0},
    ])

    monkeypatch.setattr(sp, "_fetch_unscored",
                        lambda _engine: unscored.copy(), raising=True)
    monkeypatch.setattr(sp, "_fetch_actuals",
                        lambda _engine, _skus, _weeks: actuals.copy(), raising=True)
    monkeypatch.setattr(sp, "get_engine", lambda: object(), raising=True)

    captured = _RecordingSession()

    @contextmanager
    def _scope():
        yield captured

    monkeypatch.setattr(sp, "session_scope", _scope, raising=True)
    sp.run_scoring(dry_run=False)

    update_calls = [c for c in captured.calls if "UPDATE prediction_log" in c[0]]
    upsert_calls = [c for c in captured.calls if "INSERT INTO model_performance_live" in c[0]]
    assert update_calls, f"expected an UPDATE on prediction_log, got: {[c[0][:60] for c in captured.calls]}"
    assert upsert_calls, f"expected an INSERT on model_performance_live, got: {[c[0][:60] for c in captured.calls]}"

    # The update payload must hold one row per joined prediction.
    update_payload = update_calls[0][1]
    assert isinstance(update_payload, list)
    assert len(update_payload) == 2
    assert {row["id"] for row in update_payload} == {1, 2}
    assert update_payload[0]["actual_units"] == 100.0

    # The upsert payload must contain computed metrics for the (model, category) pair.
    upsert_payload = upsert_calls[0][1]
    assert isinstance(upsert_payload, list)
    assert len(upsert_payload) == 1
    metric_row = upsert_payload[0]
    assert metric_row["model_version"] == "forecaster_milk_v1"
    assert metric_row["category"] == "Milk"
    assert metric_row["samples"] == 2
    # MAE = (|100 - 100| + |90 - 110|) / 2 = 10.0
    assert metric_row["live_mae"] == pytest.approx(10.0)
    # MAPE = ((0/100) + (20/90)) / 2 ≈ 0.1111
    assert metric_row["live_mape"] == pytest.approx((0.0 + 20.0 / 90.0) / 2.0, rel=1e-6)


def test_score_predictions_no_unscored_rows_returns_silently(monkeypatch) -> None:
    """Idempotency: when no rows need scoring, run_scoring must short-circuit
    without touching session_scope (re-running the same day stays a no-op)."""
    from scripts import score_predictions as sp

    monkeypatch.setattr(sp, "_fetch_unscored",
                        lambda _engine: pd.DataFrame(), raising=True)
    monkeypatch.setattr(sp, "get_engine", lambda: object(), raising=True)

    @contextmanager
    def _exploding_scope():
        raise AssertionError("session_scope must not be entered when no unscored rows")
        yield  # pragma: no cover

    monkeypatch.setattr(sp, "session_scope", _exploding_scope, raising=True)
    # Must not raise.
    sp.run_scoring(dry_run=False)


# =================== Cloud Run / scheduler IaC sanity ======================


def test_scheduler_script_creates_both_jobs_paused() -> None:
    """Both Cloud Scheduler jobs must be created with the --paused flag.

    Per Phase D spec: schedules are defined in IaC but never auto-fire — a
    human must explicitly resume them. Drop of `--paused` would silently turn
    on production retraining.
    """
    from pathlib import Path

    script_path = (Path(__file__).resolve().parent.parent
                   / "scripts" / "cloud_jobs" / "scheduler.gcloud.sh")
    assert script_path.exists(), f"expected scheduler script at {script_path}"
    text = script_path.read_text(encoding="utf-8")

    # Both job creates exist.
    assert "gcloud scheduler jobs create http fmcg-retrain-weekly" in text
    assert "gcloud scheduler jobs create http fmcg-score-daily" in text

    # Find each create-block and verify --paused appears within it (before the
    # next blank line / next gcloud command).
    for job_name in ("fmcg-retrain-weekly", "fmcg-score-daily"):
        idx = text.index(f"gcloud scheduler jobs create http {job_name}")
        # Take the next 800 chars (well within a single create command).
        block = text[idx:idx + 800]
        # Strip everything after the next standalone gcloud invocation.
        next_create = block.find("\ngcloud scheduler jobs create http", 1)
        if next_create > 0:
            block = block[:next_create]
        assert "--paused" in block, (
            f"job {job_name} must be created with --paused flag — IaC must not "
            f"auto-activate production schedules"
        )


def test_cloud_run_yamls_bind_secrets_by_reference() -> None:
    """Cloud Run Job YAMLs must reference Secret Manager via `secretKeyRef` —
    NEVER inline DB_PASSWORD or API_KEY values."""
    from pathlib import Path

    base = Path(__file__).resolve().parent.parent / "scripts" / "cloud_jobs"
    for fname in ("retrain_job.yaml", "score_job.yaml"):
        text = (base / fname).read_text(encoding="utf-8")
        assert "secretKeyRef" in text, f"{fname} must use secretKeyRef for secrets"
        # Defensive grep: no inline raw `password:` value (anything not a key ref).
        # The YAML may define `name: db-password` — that's the secret resource
        # name reference inside `secretKeyRef`, which is fine.
        for forbidden_pattern in (
            'value: "fmcg_password"', 'value: "test_password"',
            'value: "secret"', "DB_PASSWORD: '",
        ):
            assert forbidden_pattern not in text, (
                f"{fname} appears to inline a literal secret: {forbidden_pattern!r}"
            )
