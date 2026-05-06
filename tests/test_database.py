"""Database / schema unit tests.

These tests exercise pure logic that doesn't require a live PostgreSQL — schema
SQL parsing, allow-list table validation, and the DDL constraint surface.
A live integration test requires PostgreSQL; we verify schema text properties.
"""
from __future__ import annotations

from pathlib import Path

import pytest

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "src" / "database" / "schema.sql"


@pytest.fixture(scope="module")
def schema_sql() -> str:
    """Read the shipped schema.sql once per module."""
    return SCHEMA_PATH.read_text(encoding="utf-8")


def test_schema_creates_all_required_tables(schema_sql: str) -> None:
    """Every required table appears in CREATE TABLE form."""
    required = {
        "products",
        "daily_sales",
        "weekly_features",
        "enrichment_features",
        "demand_forecasts",
        "batch_predictions",
    }
    for table in required:
        assert f"CREATE TABLE {table}" in schema_sql, table


def test_schema_does_not_clamp_units_sold_to_non_negative(schema_sql: str) -> None:
    """Negative units_sold is valid (returns) — must NOT be CHECK-clamped."""
    assert "CHECK (units_sold >= 0)" not in schema_sql
    assert "CHECK (delivered_qty >= 0)" not in schema_sql


def test_schema_constrains_lifecycle_stage(schema_sql: str) -> None:
    """`lifecycle_stage` must be constrained to Growth/Mature/Decline."""
    assert "'Growth'" in schema_sql
    assert "'Mature'" in schema_sql
    assert "'Decline'" in schema_sql


def test_schema_constrains_price_unit_positive(schema_sql: str) -> None:
    """`price_unit > 0` constraint must be present in daily_sales."""
    assert "price_unit > 0" in schema_sql


def test_schema_indexes_on_critical_columns(schema_sql: str) -> None:
    """Indexes on sku, date/week, channel/region are present."""
    assert "CREATE INDEX" in schema_sql
    assert "idx_daily_sales_sku" in schema_sql
    assert "idx_daily_sales_date" in schema_sql
    assert "idx_weekly_features_week" in schema_sql


def test_loaders_table_allowlist_rejects_unknown_table(monkeypatch) -> None:
    """`upsert_dataframe` must reject any table not in its allow-list."""
    import pandas as pd
    from src.etl import loaders
    with pytest.raises(ValueError):
        loaders.upsert_dataframe("attacker_table", pd.DataFrame([{"a": 1}]), ["a"])


def test_loaders_empty_dataframe_returns_zero() -> None:
    """An empty frame should short-circuit to a zero-row insert."""
    import pandas as pd
    from src.etl import loaders
    assert loaders.upsert_dataframe("products", pd.DataFrame(), ["sku"]) == 0


def test_connection_module_uses_pool_pre_ping() -> None:
    """Engine creation should enable `pool_pre_ping` for production safety."""
    from src.database import connection
    source = (Path(connection.__file__)).read_text(encoding="utf-8")
    assert "pool_pre_ping=True" in source


# ---------------- Phase B regression tests ----------------------------------


def test_split_sql_statements_yields_first_drop_after_header_comment() -> None:
    """Phase B1: header `--` comments must NOT swallow the first DROP statement.

    Pre-fix `_split_sql_statements` skipped any chunk whose first stripped line
    started with `--`, so a script beginning with a header comment would lose
    its first DROP. After the fix the regex strips line comments globally
    BEFORE splitting on `;`.
    """
    from src.database.init_db import _split_sql_statements

    script = (
        "-- FMCG schema header comment\n"
        "-- Idempotent: re-runnable\n"
        "DROP TABLE IF EXISTS first_table CASCADE;\n"
        "DROP TABLE IF EXISTS second_table CASCADE;\n"
        "CREATE TABLE first_table (id INT);\n"
    )
    statements = list(_split_sql_statements(script))
    assert statements, "expected at least one statement after stripping comments"
    assert statements[0].lstrip().startswith("DROP TABLE"), (
        f"first non-blank statement should be a DROP, got: {statements[0]!r}"
    )
    drop_count = sum(1 for stmt in statements if "DROP TABLE" in stmt)
    assert drop_count == 2, (
        f"expected 2 DROP TABLE statements, got {drop_count} — header comment "
        f"may still be swallowing one"
    )


def test_split_sql_statements_against_real_schema_emits_six_drops() -> None:
    """Phase B1 (integration): the shipped schema.sql must yield 6 DROPs.

    schema.sql begins with two `--` header lines followed by 6 DROP TABLE
    statements (one per table). The pre-B1 splitter would have dropped the
    first of those DROPs, breaking idempotency on re-runs.
    """
    from src.database.init_db import _split_sql_statements

    schema_text = SCHEMA_PATH.read_text(encoding="utf-8")
    statements = list(_split_sql_statements(schema_text))
    drops = [stmt for stmt in statements if "DROP TABLE" in stmt]
    assert len(drops) == 6, (
        f"schema.sql should yield 6 DROP statements but produced {len(drops)}: "
        f"{[d[:60] for d in drops]}"
    )
