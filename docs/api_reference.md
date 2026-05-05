# API Reference

All routes are mounted under `/api/v1`.

Authentication: every route except `/api/v1/health`, `/docs`, `/redoc`,
and `/openapi.json` requires the `X-API-Key` header.

Rate limit: 100 requests/minute per API key (configurable via
`API_RATE_LIMIT_PER_MIN`).

Error envelope:

```json
{
  "error": "string",
  "detail": "string",
  "request_id": "string"
}
```

The `X-Request-ID` response header echoes the correlation ID for log
correlation.

---

## GET /api/v1/health

Liveness/readiness probe. **Public** (no API key required).

Response — `200 OK`:

```json
{
  "db_status": "ok",
  "model_status": "ok",
  "uptime_seconds": 1234.5,
  "version": "1.0.0",
  "row_counts": {
    "products": 30,
    "daily_sales": 190757,
    "weekly_features": 31027,
    "enrichment_features": 1349,
    "demand_forecasts": 0,
    "batch_predictions": 0
  }
}
```

Example:

```bash
curl http://localhost:8000/api/v1/health
```

---

## GET /api/v1/forecast/{sku}

Forecast weekly demand for a single SKU at the chosen channel/region.

Path parameter:

- `sku` — must match `^[A-Z]{2}-\d{3}$` (e.g. `MI-006`)

Query parameters:

- `channel` — `Retail` | `Discount` | `E-commerce`
- `region` — `PL-Central` | `PL-North` | `PL-South`
- `horizon_weeks` — integer 1–12 (default 4)

Response — `200 OK`, `ForecastResult`:

```json
{
  "sku": "MI-006",
  "channel": "Retail",
  "region": "PL-Central",
  "weeks": ["2025-01-06", "2025-01-13", "2025-01-20", "2025-01-27"],
  "predicted_units": [120.4, 118.3, 121.0, 119.7],
  "confidence_lower": [102.1, 100.4, 103.5, 101.8],
  "confidence_upper": [138.7, 136.2, 138.5, 137.6],
  "model_version": "forecaster_milk_v1"
}
```

Errors:

- `404` — unknown SKU or no recent history
- `503` — forecast model unavailable for the SKU's category

Example:

```bash
curl -H "X-API-Key: $API_KEY" \
  "http://localhost:8000/api/v1/forecast/MI-006?channel=Retail&region=PL-Central&horizon_weeks=4"
```

---

## GET /api/v1/forecast/category/{category}

Forecast every SKU in a category for the same channel/region.

Path parameter:

- `category` — `Milk` | `Yogurt` | `ReadyMeal` | `Juice` | `SnackBar`

Same query parameters as `/forecast/{sku}`. Returns `list[ForecastResult]`.

```bash
curl -H "X-API-Key: $API_KEY" \
  "http://localhost:8000/api/v1/forecast/category/Milk?channel=Retail&region=PL-Central"
```

---

## POST /api/v1/forecast/batch

Run inference on a parquet batch already staged inside `RAW_DATA_DIR`.

Query parameter:

- `parquet_filename` — basename only (rejects path traversal)

Response — `200 OK`, `BatchPredictionResult`:

```json
{
  "batch_id": "batch-20250106-120000-a1b2c3",
  "predictions": [ ...ForecastResult... ],
  "created_at": "2025-01-06T12:00:00Z"
}
```

Example:

```bash
curl -X POST -H "X-API-Key: $API_KEY" \
  "http://localhost:8000/api/v1/forecast/batch?parquet_filename=batch_MI-006_2025-01-06.parquet"
```

---

## GET /api/v1/analytics/sales-summary

Aggregate sales between two dates with optional category filter.

Query parameters:

- `start_date`, `end_date` — `YYYY-MM-DD` (inclusive)
- `category` — optional category filter

Response — `200 OK`, `SalesSummary` with totals and a per-category breakdown.

```bash
curl -H "X-API-Key: $API_KEY" \
  "http://localhost:8000/api/v1/analytics/sales-summary?start_date=2024-01-01&end_date=2024-12-31"
```

---

## GET /api/v1/analytics/top-products

Top N products ranked by `units_sold` or `revenue`.

Query parameters:

- `n` — integer 1–100 (default 10)
- `metric` — `units_sold` | `revenue` (default `units_sold`)

Returns `list[TopProduct]`.

```bash
curl -H "X-API-Key: $API_KEY" \
  "http://localhost:8000/api/v1/analytics/top-products?n=10&metric=revenue"
```

---

## GET /api/v1/analytics/inventory-risk

SKUs whose stock cover is shorter than `threshold_days`.

Query parameter:

- `threshold_days` — integer 1–30 (default 7)

Returns `list[InventoryRiskItem]`.

```bash
curl -H "X-API-Key: $API_KEY" \
  "http://localhost:8000/api/v1/analytics/inventory-risk?threshold_days=10"
```

---

## GET /api/v1/analytics/category-trends

Total monthly units sold per category over the trailing N months.

Query parameter:

- `months` — integer 1–24 (default 6)

Returns a list of `{category, month, total_units_sold, delta_pct}` records.

---

## GET /api/v1/analytics/lifecycle-distribution

Counts of SKUs per Growth/Mature/Decline stage.

Returns `LifecycleDistribution`:

```json
{ "growth": 5, "mature": 20, "decline": 5 }
```

---

## GET /api/v1/analytics/channel-comparison

Per-channel KPIs (units sold, average price, promo lift).

Returns `list[ChannelComparisonRow]`.

---

## Validation rules reference

| Field            | Constraint                                           |
|------------------|------------------------------------------------------|
| `sku`            | regex `^[A-Z]{2}-\d{3}$`                             |
| `channel`        | one of three literals                                |
| `region`         | one of three literals                                |
| `category`       | one of five literals                                 |
| `horizon_weeks`  | integer in [1, 12]                                   |
| `n` (top-prods)  | integer in [1, 100]                                  |
| `threshold_days` | integer in [1, 30]                                   |
| `months`         | integer in [1, 24]                                   |
| `start_date`     | ≤ `end_date` (otherwise `422 Unprocessable Entity`)  |

All failed validations return `422` with details describing the offending
field. Authentication failures return `401`. Rate-limit hits return `429`.
Internal errors return `500` with the standard error envelope.
