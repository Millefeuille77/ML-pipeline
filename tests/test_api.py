"""API route tests (auth, rate limit, validation, surface).

These tests build a minimal FastAPI app composed from the real middleware
stack but mock out the database engine so they don't require PostgreSQL.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

os.environ.setdefault("API_KEY", "test_api_key_value")
os.environ.setdefault("DB_PASSWORD", "test_password")


class _FakeResult:
    """Stand-in for a SQLAlchemy result row/proxy used by the routes."""

    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def scalar(self) -> Any:
        if isinstance(self._payload, list):
            if not self._payload:
                # WHY: real Postgres returns None from `MAX(...)` / `SELECT ...`
                # against an empty table. Empty list payload mirrors that.
                return None
            first = self._payload[0]
            return first[0] if isinstance(first, tuple) else first
        return self._payload

    def fetchone(self) -> Any:
        if isinstance(self._payload, list) and self._payload:
            return self._payload[0]
        return None

    def fetchall(self) -> Any:
        return self._payload if isinstance(self._payload, list) else []

    def mappings(self):
        rows = self._payload if isinstance(self._payload, list) else []
        normalized = []
        for row in rows:
            if isinstance(row, dict):
                normalized.append(row)
            elif isinstance(row, tuple):
                normalized.append({f"col{i}": v for i, v in enumerate(row)})
        return _FakeMappings(normalized)


class _FakeMappings:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def all(self) -> list[dict[str, Any]]:
        return list(self._rows)


class _FakeConnection:
    """Stand-in connection that returns canned payloads keyed by SQL fragment."""

    def __init__(self, payloads: dict[str, Any]) -> None:
        self._payloads = payloads

    def execute(self, statement, params=None):  # noqa: ARG002
        sql_text = str(getattr(statement, "text", statement))
        for fragment, payload in self._payloads.items():
            if fragment in sql_text:
                return _FakeResult(payload)
        return _FakeResult([])

    def __enter__(self) -> "_FakeConnection":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def close(self) -> None:
        return None


class _FakeEngine:
    def __init__(self, payloads: dict[str, Any]) -> None:
        self._payloads = payloads

    def connect(self) -> _FakeConnection:
        return _FakeConnection(self._payloads)


@contextmanager
def _patched_engine(monkeypatch: pytest.MonkeyPatch, payloads: dict[str, Any]):
    """Patch every `get_engine` call site to return the fake engine."""
    fake = _FakeEngine(payloads)
    targets = [
        "src.database.connection.get_engine",
        "src.api.routes.health.get_engine",
        "src.api.routes.forecast.get_engine",
        "src.api.routes.analytics.get_engine",
    ]
    for target in targets:
        monkeypatch.setattr(target, lambda fake=fake: fake, raising=True)
    yield fake


def _build_test_app() -> FastAPI:
    """Compose a minimal FastAPI app mirroring `src/main.py` middleware order."""
    from src.api.middleware.auth import ApiKeyAuthMiddleware
    from src.api.middleware.error_handler import ErrorHandlerMiddleware
    from src.api.middleware.rate_limiter import RateLimitMiddleware
    from src.api.routes import analytics, forecast, health
    application = FastAPI()
    application.add_middleware(RateLimitMiddleware)
    application.add_middleware(ApiKeyAuthMiddleware)
    application.add_middleware(ErrorHandlerMiddleware)
    application.include_router(health.router, prefix="/api/v1")
    application.include_router(forecast.router, prefix="/api/v1")
    application.include_router(analytics.router, prefix="/api/v1")
    return application


@pytest.fixture
def fake_health_payloads():
    """Canned DB responses for the health endpoint.

    Phase D additions: prediction_log / model_lineage / model_performance_live
    row counts plus the three helper queries (MAX(completed_at), live MAPE
    aggregation, stale-model lookup) all map onto empty payloads. The
    `_FakeResult.scalar()` empty-list short-circuit returns None, mirroring
    real Postgres behavior on empty tables, so the route helpers must take
    their default branches and produce safe defaults.
    """
    # WHY: dict iteration order is insertion order, and `_FakeConnection.execute`
    # returns the first matching fragment. More specific fragments must come
    # before generic table-name fragments so the wrong query doesn't grab the
    # wrong canned payload. The Phase D helpers query MAX/AVG/DISTINCT on
    # tables that ALSO have a COUNT(*) row in the fixture.
    return {
        "SELECT 1": [(1,)],
        # Phase D MAX(completed_at) → None scalar (empty table).
        "MAX(completed_at) FROM model_lineage": [],
        # Phase D AVG(live_mape) GROUP BY category → empty rows.
        "AVG(live_mape)": [],
        # Phase D stale-model lookup — empty rows → empty list.
        "WHERE category NOT IN": [],
        "FROM products": [(30,)],
        "FROM daily_sales": [(190757,)],
        "FROM weekly_features": [(31027,)],
        "FROM enrichment_features": [(1349,)],
        "FROM demand_forecasts": [(0,)],
        "FROM batch_predictions": [(0,)],
        "FROM prediction_log": [(0,)],
        "FROM model_lineage": [(0,)],
        "FROM model_performance_live": [(0,)],
    }


def test_health_returns_200_without_api_key(monkeypatch, fake_health_payloads) -> None:
    """`/api/v1/health` must be reachable without an API key."""
    with _patched_engine(monkeypatch, fake_health_payloads):
        client = TestClient(_build_test_app())
        response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["db_status"] == "ok"
    assert "row_counts" in body


def test_request_without_api_key_is_rejected(monkeypatch, fake_health_payloads) -> None:
    """Protected endpoints reject calls missing `X-API-Key`."""
    with _patched_engine(monkeypatch, fake_health_payloads):
        client = TestClient(_build_test_app())
        response = client.get("/api/v1/analytics/lifecycle-distribution")
    assert response.status_code == 401
    assert response.json()["error"] == "unauthorized"


def test_request_with_wrong_api_key_is_rejected(monkeypatch, fake_health_payloads) -> None:
    """Wrong API key returns 401."""
    with _patched_engine(monkeypatch, fake_health_payloads):
        client = TestClient(_build_test_app())
        response = client.get(
            "/api/v1/analytics/lifecycle-distribution",
            headers={"X-API-Key": "definitely_wrong"},
        )
    assert response.status_code == 401


def test_lifecycle_distribution_returns_counts(monkeypatch, test_api_key) -> None:
    """Lifecycle endpoint returns Growth/Mature/Decline counts."""
    payloads = {
        "FROM weekly_features": [
            {"lifecycle_stage": "Growth", "sku_count": 5},
            {"lifecycle_stage": "Mature", "sku_count": 20},
            {"lifecycle_stage": "Decline", "sku_count": 5},
        ],
    }
    with _patched_engine(monkeypatch, payloads):
        client = TestClient(_build_test_app())
        response = client.get(
            "/api/v1/analytics/lifecycle-distribution",
            headers={"X-API-Key": test_api_key},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["growth"] == 5
    assert body["mature"] == 20
    assert body["decline"] == 5


def test_top_products_validation_rejects_n_above_limit(monkeypatch, test_api_key) -> None:
    """`n=101` must yield 422 (Query bound is le=100)."""
    with _patched_engine(monkeypatch, {}):
        client = TestClient(_build_test_app())
        response = client.get(
            "/api/v1/analytics/top-products?n=101",
            headers={"X-API-Key": test_api_key},
        )
    assert response.status_code == 422


def test_sales_summary_invalid_date_range_rejected(monkeypatch, test_api_key) -> None:
    """`start_date > end_date` must yield 422."""
    with _patched_engine(monkeypatch, {}):
        client = TestClient(_build_test_app())
        response = client.get(
            "/api/v1/analytics/sales-summary",
            params={"start_date": "2024-12-31", "end_date": "2024-01-01"},
            headers={"X-API-Key": test_api_key},
        )
    assert response.status_code == 422


def test_forecast_invalid_sku_pattern_rejected(monkeypatch, test_api_key) -> None:
    """SKU not matching `XX-NNN` must yield 422."""
    with _patched_engine(monkeypatch, {}):
        client = TestClient(_build_test_app())
        response = client.get(
            "/api/v1/forecast/INVALID",
            params={"channel": "Retail", "region": "PL-Central"},
            headers={"X-API-Key": test_api_key},
        )
    assert response.status_code == 422


def test_forecast_missing_sku_returns_404(monkeypatch, test_api_key) -> None:
    """Unknown SKU yields 404 from `_resolve_category`."""
    payloads = {"FROM products WHERE sku": []}
    with _patched_engine(monkeypatch, payloads):
        client = TestClient(_build_test_app())
        response = client.get(
            "/api/v1/forecast/MI-999",
            params={"channel": "Retail", "region": "PL-Central"},
            headers={"X-API-Key": test_api_key},
        )
    assert response.status_code == 404


def test_rate_limit_exceeded_returns_429(monkeypatch, fake_health_payloads, test_api_key) -> None:
    """A single low-cap limiter triggers 429 after threshold."""
    monkeypatch.setenv("API_RATE_LIMIT_PER_MIN", "3")
    from config import settings as settings_module
    settings_module.get_settings.cache_clear()
    with _patched_engine(monkeypatch, fake_health_payloads):
        client = TestClient(_build_test_app())
        results = [
            client.get(
                "/api/v1/analytics/lifecycle-distribution",
                headers={"X-API-Key": test_api_key},
            ).status_code
            for _ in range(5)
        ]
    settings_module.get_settings.cache_clear()
    assert results.count(429) >= 1


def test_error_handler_returns_structured_json(monkeypatch, test_api_key) -> None:
    """Unhandled exceptions are converted to a JSON error envelope."""
    def _boom(*_args, **_kwargs):
        raise RuntimeError("simulated")
    monkeypatch.setattr("src.api.routes.health.check_health", _boom, raising=True)
    with _patched_engine(monkeypatch, {}):
        client = TestClient(_build_test_app(), raise_server_exceptions=False)
        response = client.get("/api/v1/health")
    assert response.status_code == 500
    body = response.json()
    assert body["error"] == "internal_error"
    assert "request_id" in body


# ---------------- HIGH-1 fix: IP-bucket rate-limit on /health ---------------

def test_health_rate_limited_when_unauthenticated(
    monkeypatch, fake_health_payloads
) -> None:
    """101 rapid unauthenticated `/health` calls share an IP bucket → 429.

    With `API_RATE_LIMIT_PER_MIN=4`, the 5th call must be rejected with 429.
    Pre-fix this would have been impossible because the limiter only bucketed
    by API key, leaving auth-exempt routes unlimited.
    """
    monkeypatch.setenv("API_RATE_LIMIT_PER_MIN", "4")
    from config import settings as settings_module
    settings_module.get_settings.cache_clear()
    with _patched_engine(monkeypatch, fake_health_payloads):
        client = TestClient(_build_test_app())
        statuses = [client.get("/api/v1/health").status_code for _ in range(6)]
    settings_module.get_settings.cache_clear()
    assert 429 in statuses, f"expected at least one 429 in {statuses}"
    assert statuses[-1] == 429
    assert statuses[0] == 200  # first call still allowed


# ---------------- forecast happy path ---------------------------------------

def test_get_forecast_happy_path(monkeypatch, test_api_key) -> None:
    """`GET /api/v1/forecast/MI-006?...` returns a populated ForecastResult body."""
    from datetime import date as date_type
    from src.api import schemas
    from src.api.routes import forecast as forecast_route

    deterministic = schemas.ForecastResult(
        sku="MI-006",
        channel="Retail",
        region="PL-Central",
        weeks=[date_type(2025, 1, 6), date_type(2025, 1, 13),
               date_type(2025, 1, 20), date_type(2025, 1, 27)],
        predicted_units=[100.0, 102.0, 104.0, 106.0],
        confidence_lower=[90.0, 92.0, 94.0, 96.0],
        confidence_upper=[110.0, 112.0, 114.0, 116.0],
        model_version="forecaster_milk_v1",
    )

    class _StubInfo:
        name = "forecaster_milk"
        version = 1

    monkeypatch.setattr(
        forecast_route, "_resolve_category", lambda sku: "Milk", raising=True
    )
    monkeypatch.setattr(
        forecast_route,
        "_load_recent_history",
        lambda sku, channel, region: __import__("pandas").DataFrame(
            [{"week": date_type(2024, 12, 30), "units_sold": 100.0}]
        ),
        raising=True,
    )
    monkeypatch.setattr(
        forecast_route, "_safe_get_model", lambda category: (object(), _StubInfo()),
        raising=True,
    )
    monkeypatch.setattr(
        forecast_route, "predict_horizon",
        lambda bundle, sku, channel, region, history, horizon: deterministic,
        raising=True,
    )

    with _patched_engine(monkeypatch, {}):
        client = TestClient(_build_test_app())
        response = client.get(
            "/api/v1/forecast/MI-006",
            params={"channel": "Retail", "region": "PL-Central", "horizon_weeks": 4},
            headers={"X-API-Key": test_api_key},
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["sku"] == "MI-006"
    assert body["channel"] == "Retail"
    assert len(body["predicted_units"]) == 4
    assert body["predicted_units"][0] == 100.0


# ---------------- analytics happy paths --------------------------------------

def test_get_sales_summary_happy_path(monkeypatch, test_api_key) -> None:
    """`/analytics/sales-summary` returns 200 with non-empty body when DB has rows."""
    payloads = {
        "AS total_units": [(2500.0, 12345.67, 100)],
        "GROUP BY p.category": [
            {"category": "Milk", "units_sold": 1500.0},
            {"category": "Yogurt", "units_sold": 1000.0},
        ],
    }
    with _patched_engine(monkeypatch, payloads):
        client = TestClient(_build_test_app())
        response = client.get(
            "/api/v1/analytics/sales-summary",
            params={"start_date": "2024-01-01", "end_date": "2024-12-31"},
            headers={"X-API-Key": test_api_key},
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total_units_sold"] == 2500.0
    assert body["total_revenue"] == 12345.67
    assert body["record_count"] == 100
    assert len(body["breakdown"]) == 2


def test_top_products_returns_exact_count(monkeypatch, test_api_key) -> None:
    """`?n=5` must return exactly 5 items, no more, no less."""
    rows = [
        {"sku": f"MI-{idx + 1:03d}", "category": "Milk",
         "metric_value": float(100 - idx * 5)}
        for idx in range(5)
    ]
    payloads = {"GROUP BY p.sku, p.category": rows}
    with _patched_engine(monkeypatch, payloads):
        client = TestClient(_build_test_app())
        response = client.get(
            "/api/v1/analytics/top-products?n=5",
            headers={"X-API-Key": test_api_key},
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body) == 5
    assert body[0]["sku"] == "MI-001"
    assert body[0]["metric_value"] == 100.0


def test_top_products_rejects_zero(monkeypatch, test_api_key) -> None:
    """`?n=0` must yield 422 (Query bound is ge=1)."""
    with _patched_engine(monkeypatch, {}):
        client = TestClient(_build_test_app())
        response = client.get(
            "/api/v1/analytics/top-products?n=0",
            headers={"X-API-Key": test_api_key},
        )
    assert response.status_code == 422


# ---------------- auth: constant-time comparator ----------------------------

@pytest.mark.parametrize(
    "provided, expected, want",
    [
        ("abc", "abc", True),
        ("abc", "abcd", False),       # length mismatch — shortcut path
        ("abc", "abd", False),        # same length, content mismatch
        ("", "", True),
        ("Z", "z", False),            # case sensitive
        ("api_key_123", "api_key_124", False),
    ],
)
def test_safe_compare_constant_time(provided: str, expected: str, want: bool) -> None:
    """`_safe_compare` matches `hmac.compare_digest` semantics for behavior cases."""
    from src.api.middleware.auth import _safe_compare

    assert _safe_compare(provided, expected) is want


# ---------------- Phase B regression tests ----------------------------------


def test_category_trends_handles_decimal_units_sold(
    monkeypatch: pytest.MonkeyPatch, test_api_key: str
) -> None:
    """Phase B5: Postgres NUMERIC arrives as `decimal.Decimal` in pandas.

    Pre-fix: `frame.groupby(...)["total_units_sold"].pct_change() * 100.0`
    raised `TypeError: unsupported operand type(s) for *: 'decimal.Decimal' and
    'float'` because pct_change preserves Decimal objects.

    Post-fix: column is cast to float before `pct_change`. This test feeds
    Decimal-valued payloads through the fake DB and asserts a 200 response with
    numeric `delta_pct` values.
    """
    from datetime import date as date_type
    from decimal import Decimal

    payloads = {
        "GROUP BY p.category, DATE_TRUNC": [
            {"category": "Milk",   "month": date_type(2024, 12, 1),
             "total_units_sold": Decimal("1500.0000")},
            {"category": "Milk",   "month": date_type(2024, 11, 1),
             "total_units_sold": Decimal("1200.0000")},
            {"category": "Yogurt", "month": date_type(2024, 12, 1),
             "total_units_sold": Decimal("800.0000")},
            {"category": "Yogurt", "month": date_type(2024, 11, 1),
             "total_units_sold": Decimal("600.0000")},
        ],
    }
    with _patched_engine(monkeypatch, payloads):
        client = TestClient(_build_test_app())
        response = client.get(
            "/api/v1/analytics/category-trends",
            params={"months": 2},
            headers={"X-API-Key": test_api_key},
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert isinstance(body, list)
    assert len(body) == 4
    # Every row must have a numeric (float / int / None) `delta_pct`. If the
    # cast were missing, we'd see a 500 from the route's TypeError.
    for row in body:
        assert "delta_pct" in row
        delta = row["delta_pct"]
        assert delta is None or isinstance(delta, (int, float)), (
            f"delta_pct should be numeric-or-null, got {delta!r} ({type(delta).__name__})"
        )
    # Sanity: the second row in each category (sorted by month asc) is the
    # later-month value, so its pct_change is non-null. With 1500 vs 1200 the
    # delta is +25%.
    milk_rows = [row for row in body if row["category"] == "Milk"]
    milk_rows.sort(key=lambda row: row["month"])
    assert milk_rows[0]["delta_pct"] is None  # first row in group has no prior
    assert milk_rows[1]["delta_pct"] == pytest.approx(25.0, rel=1e-3)


def test_category_trends_empty_db_returns_empty_list(
    monkeypatch: pytest.MonkeyPatch, test_api_key: str
) -> None:
    """Phase B5 invariant: empty DB still works (early-return short-circuit
    before the pct_change branch even runs)."""
    payloads = {"GROUP BY p.category, DATE_TRUNC": []}
    with _patched_engine(monkeypatch, payloads):
        client = TestClient(_build_test_app())
        response = client.get(
            "/api/v1/analytics/category-trends",
            params={"months": 6},
            headers={"X-API-Key": test_api_key},
        )
    assert response.status_code == 200
    assert response.json() == []


# =================== Phase D regression tests ==============================


def _stub_forecast_dependencies(monkeypatch, *, horizon_weeks: int = 4):
    """Patch forecast-route helpers and return the deterministic ForecastResult
    that the route will hand the BackgroundTask.

    Centralizing the stub keeps the Phase D tests readable and lets each
    individual test override only what it cares about (e.g. the session_scope
    side-effect for failure / idempotency cases).
    """
    from datetime import date as date_type
    from src.api import schemas
    from src.api.routes import forecast as forecast_route

    weeks = [date_type(2025, 1, 6 + 7 * i) for i in range(horizon_weeks)]
    deterministic = schemas.ForecastResult(
        sku="MI-006",
        channel="Retail",
        region="PL-Central",
        weeks=weeks,
        predicted_units=[100.0 + i for i in range(horizon_weeks)],
        confidence_lower=[90.0 + i for i in range(horizon_weeks)],
        confidence_upper=[110.0 + i for i in range(horizon_weeks)],
        model_version="forecaster_milk_v1",
    )

    class _StubInfo:
        name = "forecaster_milk"
        version = 1

    import pandas as _pd
    monkeypatch.setattr(forecast_route, "_resolve_category",
                        lambda sku: "Milk", raising=True)
    monkeypatch.setattr(
        forecast_route, "_load_recent_history",
        lambda sku, channel, region: _pd.DataFrame(
            [{"week": date_type(2024, 12, 30), "units_sold": 100.0}]
        ),
        raising=True,
    )
    monkeypatch.setattr(
        forecast_route, "_safe_get_model",
        lambda category: (object(), _StubInfo()), raising=True,
    )
    monkeypatch.setattr(
        forecast_route, "predict_horizon",
        lambda bundle, sku, channel, region, history, horizon: deterministic,
        raising=True,
    )
    return deterministic, forecast_route


class _CapturingSession:
    """Records every session.execute(payload) call without raising."""

    def __init__(self) -> None:
        self.calls: list[list[dict]] = []

    def execute(self, _stmt, payload):
        if isinstance(payload, list):
            self.calls.append(payload)
        elif isinstance(payload, dict):
            self.calls.append([payload])
        return self

    def commit(self) -> None:  # pragma: no cover
        pass

    def rollback(self) -> None:  # pragma: no cover
        pass

    def close(self) -> None:  # pragma: no cover
        pass


def _install_capturing_session(monkeypatch, target_module) -> _CapturingSession:
    """Replace `session_scope` inside `target_module` with a context manager
    yielding a single shared `_CapturingSession`. Returns the session so the
    caller can inspect `.calls`.
    """
    from contextlib import contextmanager

    captured = _CapturingSession()

    @contextmanager
    def _scope():
        yield captured

    monkeypatch.setattr(target_module, "session_scope", _scope, raising=True)
    return captured


def test_forecast_route_writes_prediction_log_on_200(
    monkeypatch, test_api_key
) -> None:
    """A successful /forecast must enqueue exactly `horizon_weeks` rows
    into prediction_log via a BackgroundTask. Default horizon is 4."""
    _, forecast_route = _stub_forecast_dependencies(monkeypatch, horizon_weeks=4)
    captured = _install_capturing_session(monkeypatch, forecast_route)

    with _patched_engine(monkeypatch, {}):
        client = TestClient(_build_test_app())
        response = client.get(
            "/api/v1/forecast/MI-006",
            params={"channel": "Retail", "region": "PL-Central", "horizon_weeks": 4},
            headers={"X-API-Key": test_api_key},
        )

    assert response.status_code == 200, response.text
    # TestClient runs BackgroundTasks synchronously after the response, so by
    # the time .get() returns the executes have already fired.
    assert len(captured.calls) == 1, (
        f"expected exactly one session.execute call, got {len(captured.calls)}"
    )
    rows = captured.calls[0]
    assert len(rows) == 4, f"expected 4 rows (horizon_weeks=4), got {len(rows)}"
    first = rows[0]
    assert first["sku"] == "MI-006"
    assert first["channel"] == "Retail"
    assert first["region"] == "PL-Central"
    assert first["model_version"] == "forecaster_milk_v1"
    # request_id must be UUID-shaped (32 hex chars) — never user-controlled.
    assert isinstance(first["request_id"], str)
    assert len(first["request_id"]) >= 16


def test_forecast_route_does_not_propagate_prediction_log_failure(
    monkeypatch, test_api_key
) -> None:
    """If the BackgroundTask DB write raises SQLAlchemyError the response must
    still be 200 with the forecast body — failure is swallowed and logged."""
    from contextlib import contextmanager
    from sqlalchemy.exc import SQLAlchemyError

    _, forecast_route = _stub_forecast_dependencies(monkeypatch, horizon_weeks=4)

    @contextmanager
    def _raising_scope():
        raise SQLAlchemyError("simulated DB outage")
        yield  # pragma: no cover

    monkeypatch.setattr(forecast_route, "session_scope", _raising_scope, raising=True)

    with _patched_engine(monkeypatch, {}):
        client = TestClient(_build_test_app())
        response = client.get(
            "/api/v1/forecast/MI-006",
            params={"channel": "Retail", "region": "PL-Central", "horizon_weeks": 4},
            headers={"X-API-Key": test_api_key},
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["sku"] == "MI-006"
    assert len(body["predicted_units"]) == 4


def test_forecast_route_prediction_log_idempotent_on_duplicate_request(
    monkeypatch, test_api_key
) -> None:
    """Two identical /forecast requests with the same correlation_id must both
    succeed; the underlying SQL relies on ON CONFLICT DO NOTHING for idempotency.

    The route's SQL string is asserted to contain the conflict clause as well —
    if a future edit drops it the test fails loudly.
    """
    _, forecast_route = _stub_forecast_dependencies(monkeypatch, horizon_weeks=2)
    captured = _install_capturing_session(monkeypatch, forecast_route)

    with _patched_engine(monkeypatch, {}):
        client = TestClient(_build_test_app())
        # Same correlation header on both requests (route uses it as request_id).
        headers = {"X-API-Key": test_api_key, "X-Correlation-ID": "fixed-corr-id-12345"}
        params = {"channel": "Retail", "region": "PL-Central", "horizon_weeks": 2}
        first = client.get("/api/v1/forecast/MI-006", params=params, headers=headers)
        second = client.get("/api/v1/forecast/MI-006", params=params, headers=headers)

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    # Both background tasks must have executed (no exception).
    assert len(captured.calls) == 2

    # Verify the SQL itself uses ON CONFLICT DO NOTHING (Postgres idempotency).
    import inspect
    source = inspect.getsource(forecast_route._write_prediction_log)
    assert "ON CONFLICT" in source
    assert "DO NOTHING" in source


def test_health_includes_phase_d_fields_with_safe_defaults(
    monkeypatch, fake_health_payloads
) -> None:
    """When model_lineage and model_performance_live are empty, /health must
    expose the new Phase D fields with safe defaults — no exception, no 500.
    """
    with _patched_engine(monkeypatch, fake_health_payloads):
        client = TestClient(_build_test_app())
        response = client.get("/api/v1/health")

    assert response.status_code == 200, response.text
    body = response.json()
    assert "days_since_last_retrain" in body
    assert body["days_since_last_retrain"] is None  # MAX(...) over empty → None
    assert "live_mape_by_category" in body
    assert body["live_mape_by_category"] == {}
    assert "stale_model_warning" in body
    assert body["stale_model_warning"] == []
    # Row counts include the three new tables (extension to _TABLES_TO_COUNT).
    for new_table in ("prediction_log", "model_lineage", "model_performance_live"):
        assert new_table in body["row_counts"], (
            f"row_counts missing Phase D table {new_table}"
        )
