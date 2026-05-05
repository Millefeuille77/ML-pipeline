"""Shared pytest fixtures for the FMCG platform test suite."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("API_KEY", "test_api_key_value")
os.environ.setdefault("DB_PASSWORD", "test_password")


@pytest.fixture(autouse=True)
def _isolate_settings_cache(monkeypatch):
    """Ensure each test starts with a fresh `get_settings()` cache."""
    from config import settings as settings_module
    settings_module.get_settings.cache_clear()
    yield
    settings_module.get_settings.cache_clear()


@pytest.fixture
def test_api_key() -> str:
    """Test API key set in os.environ at import time."""
    return os.environ["API_KEY"]


@pytest.fixture
def sample_daily_df() -> pd.DataFrame:
    """Ten daily fact-table rows with two SKUs, including a return."""
    rows: list[dict[str, Any]] = []
    base_date = pd.Timestamp("2024-01-01")
    for offset in range(10):
        sku = "MI-006" if offset % 2 == 0 else "YO-001"
        rows.append({
            "date": (base_date + pd.Timedelta(days=offset)).date(),
            "sku": sku,
            "brand": "BrandA" if sku == "MI-006" else "BrandB",
            "segment": "SegA" if sku == "MI-006" else "SegB",
            "category": "Milk" if sku == "MI-006" else "Yogurt",
            "channel": "Retail",
            "region": "PL-Central",
            "pack_type": "Single",
            "price_unit": 10.0 + offset * 0.1,
            "promotion_flag": offset % 2,
            "delivery_days": 2,
            "stock_available": 200.0 - offset,
            "delivered_qty": 50.0,
            "units_sold": -5.0 if offset == 4 else 25.0 + offset,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_weekly_df() -> pd.DataFrame:
    """Ten weekly modeling rows for a single (sku, channel, region)."""
    rows: list[dict[str, Any]] = []
    base_week = pd.Timestamp("2024-01-01")
    for offset in range(10):
        rows.append({
            "sku": "MI-006",
            "week": (base_week + pd.Timedelta(weeks=offset)).date(),
            "channel": "Retail",
            "region": "PL-Central",
            "category": "Milk",
            "units_sold": 100.0 + offset,
            "stock_available": 500.0,
            "promotion_flag": offset % 2,
            "price_unit": 10.0,
            "delivery_days": 2.0,
            "is_holiday_peak": 0,
            "week_number": offset + 1,
            "month": 1 if offset < 4 else 2,
            "year": 2024,
            "is_holiday_week": 0,
            "is_summer": 0,
            "is_winter": 1,
            "sku_age": 100,
            "lifecycle_stage": "Mature",
            "lag_1": 95.0 + offset,
            "lag_2": 90.0 + offset,
            "rolling_mean_4": 95.0,
            "rolling_std_4": 5.0,
            "momentum": 5.0,
            "target_next_week": 105.0 + offset,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_enriched_df() -> pd.DataFrame:
    """Five enriched-feature rows (matches `df_weekly_MI-006_enriched.csv` shape)."""
    rows: list[dict[str, Any]] = []
    base_week = pd.Timestamp("2024-01-01")
    for offset in range(5):
        rows.append({
            "sku": "MI-006",
            "week": (base_week + pd.Timedelta(weeks=offset)).date(),
            "channel": "Retail",
            "region": "PL-Central",
            "price_avg": 10.0,
            "promo_rate": 0.5,
            "stock_avg": 500.0,
            "deliveries": 1,
            "avg_temp": 5.0 + offset,
            "inflation_index": 1.05,
            "school_in_session": 1,
            "category_trend": 1.0,
            "event_score": 0.2,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def make_weekly_row():
    """Factory that returns a single weekly-row dict with overrides."""
    def _factory(sku: str = "MI-006", **overrides: Any) -> dict[str, Any]:
        defaults: dict[str, Any] = {
            "sku": sku,
            "week": pd.Timestamp("2024-01-01").date(),
            "channel": "Retail",
            "region": "PL-Central",
            "category": "Milk",
            "units_sold": 100.0,
            "stock_available": 500.0,
            "promotion_flag": 0,
            "price_unit": 10.0,
            "delivery_days": 2.0,
            "is_holiday_peak": 0,
            "week_number": 1,
            "month": 1,
            "year": 2024,
            "is_holiday_week": 0,
            "is_summer": 0,
            "is_winter": 1,
            "sku_age": 100,
            "lifecycle_stage": "Mature",
            "lag_1": 95.0,
            "lag_2": 90.0,
            "rolling_mean_4": 95.0,
            "rolling_std_4": 5.0,
            "momentum": 5.0,
            "target_next_week": 105.0,
            "price_avg": 10.0,
        }
        defaults.update(overrides)
        return defaults
    return _factory


@pytest.fixture
def synthetic_training_df(make_weekly_row) -> pd.DataFrame:
    """Larger synthetic weekly DataFrame across categories/channels/regions."""
    rng = np.random.default_rng(seed=42)
    rows = []
    base = pd.Timestamp("2022-01-03")
    categories = {
        "Milk": ["MI-006", "MI-007", "MI-008"],
        "Yogurt": ["YO-001", "YO-002", "YO-003"],
        "SnackBar": ["SB-001", "SB-002"],
    }
    channels = ["Retail", "Discount", "E-commerce"]
    regions = ["PL-Central", "PL-North", "PL-South"]
    for category, sku_list in categories.items():
        for sku in sku_list:
            for channel in channels:
                for region in regions:
                    for week_offset in range(120):
                        units = float(50 + rng.normal(0, 5) + week_offset * 0.1)
                        target = units + float(rng.normal(0, 2))
                        rows.append(make_weekly_row(
                            sku=sku,
                            week=(base + pd.Timedelta(weeks=week_offset)).date(),
                            channel=channel,
                            region=region,
                            category=category,
                            units_sold=units,
                            target_next_week=target,
                            week_number=(week_offset % 52) + 1,
                            month=((week_offset // 4) % 12) + 1,
                            year=2022 + week_offset // 52,
                        ))
    return pd.DataFrame(rows)


@pytest.fixture
def temp_model_dir(tmp_path, monkeypatch):
    """Redirect `MODEL_DIR` to a temp directory and reset registry caches."""
    target = tmp_path / "models"
    target.mkdir()
    monkeypatch.setenv("MODEL_DIR", str(target))
    from config import settings as settings_module
    settings_module.get_settings.cache_clear()
    yield target
    settings_module.get_settings.cache_clear()


class _FakeSession:
    """Minimal session replicating PostgreSQL ON CONFLICT upsert semantics in-memory."""

    def __init__(self, store: dict[tuple, dict[str, Any]], conflict_cols: list[str]) -> None:
        self._store = store
        self._conflict_cols = conflict_cols
        self.executed_sql: list[str] = []

    def execute(self, statement: Any, payload: Any = None) -> "_FakeSession":
        sql = str(getattr(statement, "text", statement))
        self.executed_sql.append(sql)
        records = payload if isinstance(payload, list) else ([payload] if isinstance(payload, dict) else [])
        for record in records:
            key = tuple(record[col] for col in self._conflict_cols)
            self._store[key] = dict(record)
        return self

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None

    def close(self) -> None:
        return None


@pytest.fixture
def fake_upsert_store():
    """Shared in-memory key→record dict that the fake session writes to."""
    return {}


@pytest.fixture
def patch_session_scope(monkeypatch, fake_upsert_store):
    """Replace `loaders.session_scope` with a context manager backed by `_FakeSession`.

    The fixture itself is a callable: pass the conflict-column list before the
    test exercises `upsert_dataframe`. The same in-memory store backs every
    invocation so two consecutive `upsert_dataframe` calls behave like a real
    upsert (idempotent on the conflict key).
    """
    from contextlib import contextmanager

    def _install(conflict_cols: list[str]) -> dict:
        @contextmanager
        def _scope():
            yield _FakeSession(fake_upsert_store, conflict_cols)
        monkeypatch.setattr("src.etl.loaders.session_scope", _scope, raising=True)
        return fake_upsert_store

    return _install
