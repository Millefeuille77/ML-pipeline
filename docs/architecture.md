# Architecture

## Overview

The FMCG Demand Forecasting & Product Intelligence Platform is a layered
Python service that ingests real FMCG distribution data, trains per-category
demand forecasters, persists everything in PostgreSQL, and exposes an
authenticated REST API.

The platform is deliberately small: 12 pinned dependencies, one external
service (PostgreSQL), no Redis, no message broker, no cloud-vendor
lock-in.

## Component diagram

```
+---------------------+      +--------------------+      +--------------------+
|  data/raw/*.csv     |      |  PostgreSQL 16     |      |  data/models/*.jl  |
|  data/raw/*.parquet | ---> |  6 tables          | <--- |  per-category GBR  |
+---------------------+      +--------------------+      +--------------------+
          |                            |                          |
          v                            v                          v
+--------------------------------------------------------------------------+
|                          src/etl/  (extract / transform / load)          |
|                          src/database/  (connection, schema, init)       |
|                          src/models/  (forecaster, cluster, anomaly,     |
|                                        evaluation, registry)             |
|                          src/analytics/  (eda, reports)                  |
+--------------------------------------------------------------------------+
                                    |
                                    v
+--------------------------------------------------------------------------+
|                          src/api/  (FastAPI 0.115)                       |
|     middleware:  ErrorHandler  ->  ApiKeyAuth  ->  RateLimiter           |
|     routes:      /api/v1/health  /api/v1/forecast  /api/v1/analytics     |
+--------------------------------------------------------------------------+
                                    |
                                    v
                              +-----------+
                              |  Client   |
                              +-----------+
```

## Data flow

1. Raw CSV/parquet files land in `data/raw/`.
2. `python -m src.database.init_db` creates the schema and bulk-loads the
   three CSV files via `INSERT ... ON CONFLICT DO UPDATE` for idempotency.
3. `src.etl.pipeline.run_full_pipeline()` aggregates daily → weekly,
   generalizes the 9 enrichment columns from MI-006 to all 30 SKUs,
   computes lag/rolling features, and writes parquet snapshots to
   `data/processed/`.
4. `src.models.forecaster.train_per_category` fits one GBR model per
   category (Milk, Yogurt, ReadyMeal, Juice, SnackBar) plus quantile
   regression heads at q=0.1 / q=0.9 for prediction intervals. Models are
   versioned to `data/models/` via `src.models.registry`.
5. The FastAPI app serves the trained models and SQL aggregates over HTTP.

## Layer responsibilities

| Layer            | Responsibility                                                      |
|------------------|---------------------------------------------------------------------|
| `config/`        | Typed settings, structured logging, correlation IDs                  |
| `src/utils/`     | Logger accessor, validators, helper functions                        |
| `src/database/`  | SQLAlchemy engine, schema DDL, bootstrap loader                      |
| `src/etl/`       | Extractors, pure transformers, idempotent loaders, pipeline runners  |
| `src/models/`    | Feature engineering, GBR / Ridge forecasters, K-Means, anomaly rules |
| `src/analytics/` | Reusable EDA helpers and natural-language insight generation         |
| `src/api/`       | Pydantic schemas, route handlers, auth/rate-limit/error middleware   |
| `src/main.py`    | FastAPI app factory and lifespan handler                             |

## Persistence model

Six PostgreSQL tables, all defined in `src/database/schema.sql`:

| Table                | Rows (full load) | Purpose                                  |
|----------------------|------------------|------------------------------------------|
| `products`           | 30               | Master record per SKU                    |
| `daily_sales`        | 190,757          | Raw fact table                           |
| `weekly_features`    | 31,027           | Modeling table with lag/rolling features |
| `enrichment_features`| 1,349 (template) | 9 enrichment columns per SKU/week        |
| `demand_forecasts`   | grows over time  | Persisted point + interval forecasts     |
| `batch_predictions`  | grows over time  | Predictions from parquet batch jobs      |

Each table has explicit CHECK constraints for the categorical columns and
indexes on the columns used by the analytics SQL.

## Deployment topology

The platform ships as two Docker services in `docker-compose.yml`:

- `postgres` — `postgres:16-alpine`, persisted volume, healthchecked
- `app` — multi-stage Python 3.12-slim, non-root user, depends on the DB

Both services use the same `.env`. The app exposes port 8000 and is
healthchecked via `GET /api/v1/health`.

## Security boundary

- All API routes (except `/api/v1/health`, `/docs`, `/redoc`,
  `/openapi.json`) require `X-API-Key` header validation.
- The rate limiter runs in-memory with a sliding-window counter keyed by
  API key (default: 100 requests/minute).
- Unhandled exceptions are caught by the outermost middleware and returned
  as `{"error", "detail", "request_id"}` JSON bodies; full tracebacks are
  logged server-side only.
- All SQL is parameterized via SQLAlchemy `text()` with bind parameters.
- The model registry only loads files from `config.settings.MODEL_DIR`.

## Operational concerns

- Logs are emitted as single-line JSON to stdout for ingestion by any log
  shipper (Promtail, Fluent Bit, etc.).
- The correlation ID is propagated from request headers (`X-Request-ID`)
  if provided, otherwise generated server-side and surfaced back in the
  response header.
- The model registry uses versioned filenames; "latest" always resolves
  to the highest version number.
