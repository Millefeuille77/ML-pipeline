"""ETL transformer / extractor unit tests."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.etl import extractors, transformers


def test_clean_daily_data_preserves_negative_units_sold(sample_daily_df: pd.DataFrame) -> None:
    """Returns are valid: negative units_sold must NOT be filtered."""
    cleaned = transformers.clean_daily_data(sample_daily_df)
    assert (cleaned["units_sold"] < 0).any()
    assert "is_return" in cleaned.columns
    assert cleaned["is_return"].sum() >= 1


def test_clean_daily_data_raises_on_missing_columns(sample_daily_df: pd.DataFrame) -> None:
    """Dropping a required column should raise ValueError."""
    bad_frame = sample_daily_df.drop(columns=["price_unit"])
    with pytest.raises(ValueError):
        transformers.clean_daily_data(bad_frame)


def test_clean_daily_data_dates_become_dates(sample_daily_df: pd.DataFrame) -> None:
    """Date column should be coerced to `datetime.date`."""
    cleaned = transformers.clean_daily_data(sample_daily_df)
    from datetime import date as date_type
    assert isinstance(cleaned["date"].iloc[0], date_type)


def test_aggregate_to_weekly_returns_one_row_per_week(sample_daily_df: pd.DataFrame) -> None:
    """Daily → weekly aggregation should produce ≤ daily row count."""
    cleaned = transformers.clean_daily_data(sample_daily_df)
    weekly = transformers.aggregate_to_weekly(cleaned)
    assert {"sku", "week", "channel", "region", "units_sold"}.issubset(weekly.columns)
    assert len(weekly) <= len(cleaned)


def test_enrich_features_adds_all_nine_columns(
    sample_daily_df: pd.DataFrame, sample_enriched_df: pd.DataFrame
) -> None:
    """`enrich_features` must add the 9 enrichment columns to all SKUs."""
    cleaned = transformers.clean_daily_data(sample_daily_df)
    weekly = transformers.aggregate_to_weekly(cleaned)
    enriched = transformers.enrich_features(weekly, sample_enriched_df)
    expected = {
        "price_avg", "promo_rate", "stock_avg", "deliveries", "avg_temp",
        "inflation_index", "school_in_session", "category_trend", "event_score",
    }
    assert expected.issubset(enriched.columns)
    assert not enriched["avg_temp"].isna().any()


def test_compute_lag_features_fills_series_start_nans(sample_weekly_df: pd.DataFrame) -> None:
    """Series-start NaNs in lag features must be filled, not propagated."""
    enriched = sample_weekly_df.copy()
    enriched["delivered_qty"] = 0.0
    out = transformers.compute_lag_features(enriched)
    for column in ["lag_1", "lag_2", "rolling_mean_4", "rolling_std_4", "momentum"]:
        assert not out[column].isna().any(), column


def test_extract_daily_csv_file_not_found(tmp_path: Path) -> None:
    """Missing CSV path should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        extractors.extract_daily_csv(tmp_path / "missing.csv")


def test_extract_batch_parquet_round_trip(tmp_path: Path, sample_daily_df: pd.DataFrame) -> None:
    """Round-trip a parquet file through the extractor."""
    target = tmp_path / "batch.parquet"
    sample_daily_df.to_parquet(target, index=False)
    frame = extractors.extract_batch_parquet(target)
    assert len(frame) == len(sample_daily_df)
    assert "sku" in frame.columns


def test_aggregate_to_weekly_sums_units_sold(sample_daily_df: pd.DataFrame) -> None:
    """The sum of weekly units_sold must equal the sum of daily units_sold."""
    cleaned = transformers.clean_daily_data(sample_daily_df)
    weekly = transformers.aggregate_to_weekly(cleaned)
    assert pytest.approx(weekly["units_sold"].sum()) == cleaned["units_sold"].sum()


def test_compute_lag_features_lag1_equals_previous_units() -> None:
    """`lag_1[N]` must equal `units_sold[N-1]` within a single (sku, channel, region) group.

    Synthetic input: row N has units_sold = N * 10 for N in [1, 6]. After
    `compute_lag_features`, lag_1 of row N must equal the units_sold of row N-1.
    """
    base_week = pd.Timestamp("2024-01-07")
    rows = [
        {
            "sku": "MI-006",
            "week": (base_week + pd.Timedelta(weeks=offset)).date(),
            "channel": "Retail",
            "region": "PL-Central",
            "units_sold": float((offset + 1) * 10),
            "stock_available": 200.0,
            "promotion_flag": 0,
            "price_unit": 10.0,
            "delivery_days": 2.0,
        }
        for offset in range(6)
    ]
    out = transformers.compute_lag_features(pd.DataFrame(rows))
    out = out.sort_values("week").reset_index(drop=True)
    units = out["units_sold"].tolist()
    lag_1 = out["lag_1"].tolist()
    for index in range(1, len(out)):
        assert lag_1[index] == pytest.approx(units[index - 1]), (
            f"lag_1[{index}] = {lag_1[index]} but units_sold[{index - 1}] = {units[index - 1]}"
        )


def test_loader_upsert_is_idempotent(patch_session_scope) -> None:
    """Loading the same DataFrame twice must leave the row count unchanged.

    Backs `loaders.session_scope` with a fake session that mimics PostgreSQL
    ON CONFLICT semantics: keys-by-conflict-columns. After two upserts the
    in-memory store must hold exactly `len(frame)` rows.
    """
    from src.etl import loaders

    conflict_cols = ["sku"]
    store = patch_session_scope(conflict_cols)
    frame = pd.DataFrame(
        [
            {"sku": "MI-006", "brand": "BrandA", "segment": "S1",
             "category": "Milk", "pack_type": "Single"},
            {"sku": "YO-001", "brand": "BrandB", "segment": "S2",
             "category": "Yogurt", "pack_type": "Single"},
            {"sku": "SB-001", "brand": "BrandC", "segment": "S3",
             "category": "SnackBar", "pack_type": "Multipack"},
        ]
    )
    inserted_first = loaders.upsert_dataframe("products", frame, conflict_cols)
    rows_after_first = len(store)
    inserted_second = loaders.upsert_dataframe("products", frame, conflict_cols)
    rows_after_second = len(store)
    assert inserted_first == 3
    assert inserted_second == 3  # both calls write 3 rows
    assert rows_after_first == 3
    assert rows_after_second == rows_after_first  # IDEMPOTENT — count unchanged
    assert store[("MI-006",)]["category"] == "Milk"


# ---------------- Phase B regression tests ----------------------------------


def test_load_weekly_features_casts_bool_holiday_peak_to_int8(
    sample_weekly_df: pd.DataFrame, monkeypatch
) -> None:
    """Phase B2: pandas infers `is_holiday_peak` as bool from CSV but the schema
    declares SMALLINT. `load_weekly_features` must cast bool to int8 before the
    bulk insert reaches Postgres.

    Stub `_bulk_insert` to capture the frame the loader passes downstream.
    """
    from src.database import init_db

    frame = sample_weekly_df.copy()
    frame["is_holiday_peak"] = frame["is_holiday_peak"].astype(bool)
    assert frame["is_holiday_peak"].dtype == bool, "fixture precondition"

    captured: dict[str, pd.DataFrame] = {}

    def _stub_bulk_insert(table: str, frame_in: pd.DataFrame, conflict_cols) -> int:
        captured["table"] = table
        captured["frame"] = frame_in.copy()
        return len(frame_in)

    monkeypatch.setattr(init_db, "_bulk_insert", _stub_bulk_insert, raising=True)
    init_db.load_weekly_features(frame)

    assert captured["table"] == "weekly_features"
    written = captured["frame"]
    assert "is_holiday_peak" in written.columns
    assert written["is_holiday_peak"].dtype == "int8", (
        f"expected int8 cast for SMALLINT column, got {written['is_holiday_peak'].dtype}"
    )
    # Cast must preserve values (False -> 0, True -> 1)
    assert set(written["is_holiday_peak"].unique()).issubset({0, 1})


def test_load_weekly_features_leaves_non_bool_holiday_peak_alone(
    sample_weekly_df: pd.DataFrame, monkeypatch
) -> None:
    """Phase B2: when `is_holiday_peak` is already integer the loader must not
    re-cast (preserves dtype rather than churning).

    The branch guard reads `dtype == bool` — anything else falls through.
    """
    from src.database import init_db

    frame = sample_weekly_df.copy()
    frame["is_holiday_peak"] = frame["is_holiday_peak"].astype("int64")
    assert frame["is_holiday_peak"].dtype == "int64", "fixture precondition"

    captured: dict[str, pd.DataFrame] = {}

    def _stub_bulk_insert(table: str, frame_in: pd.DataFrame, conflict_cols) -> int:
        captured["frame"] = frame_in.copy()
        return len(frame_in)

    monkeypatch.setattr(init_db, "_bulk_insert", _stub_bulk_insert, raising=True)
    init_db.load_weekly_features(frame)

    written = captured["frame"]
    assert written["is_holiday_peak"].dtype == "int64", (
        f"int64 should be preserved, got {written['is_holiday_peak'].dtype}"
    )


def test_pipeline_resolve_within_rejects_path_traversal(tmp_path: Path, monkeypatch) -> None:
    """`run_batch_pipeline` must refuse `../` or absolute paths outside RAW_DATA_DIR."""
    from src.etl import pipeline

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    monkeypatch.setenv("RAW_DATA_DIR", str(raw_dir))
    from config import settings as settings_module
    settings_module.get_settings.cache_clear()

    with pytest.raises(ValueError, match="path escapes raw data dir"):
        pipeline.run_batch_pipeline(Path("../etc/passwd"))

    outside_root = tmp_path / "outside"
    outside_root.mkdir()
    outside_file = outside_root / "evil.parquet"
    outside_file.write_bytes(b"junk")
    with pytest.raises(ValueError, match="path escapes raw data dir"):
        pipeline.run_batch_pipeline(outside_file.resolve())
