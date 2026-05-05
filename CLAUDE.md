# CLAUDE.md — FMCG Demand Forecasting & Product Intelligence Platform

> **Project memory.** Every agent loads this on every turn. Keep it tight, factual, and current.

## 1. Mission
Build a portfolio-grade MVP that proves Python + SQL + AI/ML production fluency for **KNS Group** (kns.asia), an FMCG distribution conglomerate across Southeast Asia. The platform forecasts weekly SKU demand, clusters products by demand behavior, detects supply/demand anomalies, and exposes everything via a secured REST API backed by PostgreSQL.

## 2. Non-Negotiable Goals
1. **MVP first.** Every feature must run end-to-end before any optimization or polish.
2. **Lightweight & optimized.** 12 pinned dependencies, no bloat, stdlib first when reasonable.
3. **Secure by default.** API key auth, rate limiting, parameterized SQL, validated input, no leaked secrets.
4. **Reusable & scalable.** Pure functions where possible, dependency injection, clean module boundaries (no circular imports), per-category model training so the system scales horizontally.
5. **Easy to debug.** Structured JSON logging with correlation IDs, specific exception types, no silent failures.
6. **Portfolio-grade docs.** README with architecture diagram, API reference, evaluation results, and a clean EDA notebook.

## 3. Interpretation of "No Dependencies"
- **External services:** only PostgreSQL. No Redis, no Celery, no Kafka, no message broker. Rate limiter is in-memory sliding window. Model registry is file-based.
- **Python packages:** exactly the 12 listed below — pinned to exact versions in `requirements.txt`. Dev-only: `pytest`, `matplotlib`.
- **Module coupling:** routes depend on services, services depend on models/db, but no module reaches across layers. No circular imports.
- **Forbidden:** TensorFlow, PyTorch, Keras (overkill for tabular weekly data with ~150 timesteps), ORM auto-magic beyond SQLAlchemy core, anything not on the approved list.

## 4. Tech Stack (locked)
| Layer | Package | Version |
|---|---|---|
| Web framework | fastapi | 0.115.0 |
| ASGI server | uvicorn | 0.30.0 |
| ORM / SQL toolkit | sqlalchemy | 2.0.35 |
| Postgres driver | psycopg2-binary | 2.9.9 |
| DataFrames | pandas | 2.2.3 |
| Numerics | numpy | 1.26.4 |
| ML | scikit-learn | 1.5.2 |
| Validation | pydantic | 2.9.0 |
| Settings | pydantic-settings | 2.5.0 |
| .env loader | python-dotenv | 1.0.1 |
| Model serialization | joblib | 1.4.2 |
| Parquet I/O | pyarrow | 18.0.0 |

Dev-only: `pytest`, `matplotlib` (for the EDA notebook).

## 5. Real Dataset
Active copies live in `data/raw/` (also archived read-only in `context/dataset/`).

| File | Rows | Granularity | Date Range | Purpose |
|---|---|---|---|---|
| `FMCG_2022_2024.csv` | 190,757 | Daily | 2022-01-21 → 2024-12-31 | Raw fact table |
| `weekly_df_final_for_modeling.csv` | 31,027 | Weekly | 2022-02-14 → 2024-12-23 | Model training |
| `df_weekly_MI-006_enriched.csv` | 1,349 | Weekly (MI-006) | full range | Enrichment template |
| `batch_MI-006_2025-01-{06,13,20,27}.parquet` | ~50 each | Weekly batches | Jan 2025 | Batch inference simulation |

**Cardinality:** 30 SKUs · 14 brands · 13 segments · 5 categories (Milk / Yogurt / ReadyMeal / Juice / SnackBar) · 3 channels (Retail / Discount / E-commerce) · 3 regions (PL-Central / PL-North / PL-South) · 3 pack types (Multipack / Single / Carton).

**Critical data facts (read these before writing transforms or tests):**
- `units_sold` and `delivered_qty` can be **negative** — these are returns. Keep them. Do **not** filter or clamp to zero.
- `lifecycle_stage` is one of `Growth | Mature | Decline` — enforce as a CHECK constraint.
- Weekly enrichment columns (`avg_temp`, `inflation_index`, `school_in_session`, `category_trend`, `event_score`) only exist for MI-006 in the source; the ETL must generalize them to all 30 SKUs.
- Weekly file already includes `lag_1`, `lag_2`, `rolling_mean_4`, `rolling_std_4`, `momentum`, `target_next_week` — **use them, don't recompute** unless extending.

## 6. Repository Layout
```
.
├── .claude/agents/             # Operational sub-agents (Claude Code reads these)
├── CLAUDE.md                   # This file — project memory
├── README.md                   # Portfolio README (orchestrator owns)
├── Dockerfile                  # (orchestrator owns)
├── docker-compose.yml          # app + postgres (orchestrator owns)
├── requirements.txt            # 12 pinned deps (orchestrator owns)
├── .env.example                # Template — placeholders only (orchestrator owns)
├── .gitignore
│
├── config/
│   ├── settings.py             # Pydantic BaseSettings, .env-driven
│   └── logging_config.py       # Structured JSON logging w/ correlation IDs
│
├── context/                    # Read-only inputs
│   ├── Job Description_Python_SQL_AI_ML_Developer.md
│   ├── agents/                 # Original agent specs (source-of-truth docs)
│   └── dataset/                # Original dataset archive
│
├── data/
│   ├── raw/                    # Active dataset (mirror of context/dataset/)
│   ├── processed/              # ETL outputs
│   └── models/                 # joblib model artifacts + metadata JSON
│
├── docs/
│   ├── architecture.md
│   └── api_reference.md
│
├── notebooks/
│   └── eda_exploration.ipynb
│
├── scripts/                    # Optional CLI helpers (e.g., seed data, run pipeline)
│
├── src/
│   ├── main.py                 # FastAPI app entrypoint
│   ├── api/
│   │   ├── schemas.py          # ALL Pydantic request/response models
│   │   ├── routes/             # forecast.py · analytics.py · health.py
│   │   └── middleware/         # auth.py · rate_limiter.py · error_handler.py
│   ├── analytics/              # eda.py · reports.py
│   ├── database/               # connection.py · schema.sql · init_db.py
│   ├── etl/                    # extractors.py · transformers.py · loaders.py · pipeline.py
│   ├── models/                 # forecaster.py · clustering.py · anomaly.py · evaluation.py · feature_engineering.py · registry.py
│   └── utils/                  # logger.py · validators.py · helpers.py
│
└── tests/                      # conftest.py · test_database/etl/models/api.py
```

## 7. Coding Standards (QA enforces — non-negotiable)
- **Type hints** on every parameter and return type.
- **Google-style docstrings** on every public function and class.
- **Function length** ≤ 30 lines; **file length** ≤ 300 lines.
- **No magic numbers** — constants live in `config/settings.py`.
- **No hardcoded paths or credentials** — read everything via `config.settings`.
- **Parameterized SQL only** — never f-strings, `.format()`, or `+` near SQL keywords.
- **Structured logging** — `from src.utils.logger import get_logger; logger = get_logger(__name__)`. No `print()`.
- **Specific exceptions only** — no bare `except:`, no `except: pass`.
- **Imports** ordered: stdlib → third-party → local. No unused imports, no dead code.
- **Naming** descriptive — never `x`, `tmp`, `data2`.

## 8. Security Requirements
- **Auth** — `X-API-Key` validated in middleware. Only `/api/v1/health` and `/docs` exempt.
- **Rate limiting** — 100 req/min per key, in-memory sliding window (no Redis).
- **Input validation** — Pydantic on every endpoint; `Query(ge=, le=)` bounds; SKU regex constraint; no arbitrary user-controlled paths reaching `joblib.load`, `eval`, `exec`, or filesystem APIs.
- **SQL injection** — SQLAlchemy parameterized statements only. Audit grep for `f"SELECT`, `f"INSERT`, `f"UPDATE`, `f"DELETE`, `"SELECT " +`.
- **Error responses** — structured JSON `{"error", "detail", "request_id"}`. Stack traces never leaked to clients; full traceback logged server-side.
- **Secrets** — `.env.example` ships placeholders only; real `.env` is `.gitignore`d. DB connection strings never logged at INFO level.
- **Dependencies** — exact versions pinned. Only the 12 approved packages.

## 9. ML Standards
- **Validation** — walk-forward temporal splits **only**. Random shuffle on time-series is a CRITICAL bug.
- **Per-category models** — train one Gradient Boosting Regressor per category (5 total). Different demand dynamics across Milk vs SnackBar etc.
- **Confidence intervals** — quantile regression at q=0.1 (lower), q=0.5 (point), q=0.9 (upper).
- **Two-model comparison** — Gradient Boosting (primary) vs Ridge (baseline). If GBR beats Ridge by < 5% MAPE, simpler model wins.
- **Feature encoders** — fit on training data only, transform test. No leakage.
- **Model loading** — only from `config.settings.MODEL_DIR`. Never user-supplied paths.
- **Negative `units_sold`** — valid (returns). Never filter, clamp, or treat as error.
- **Persistence** — every trained model goes through `src/models/registry.py` — versioned, with metadata JSON capturing features, metrics, training rows, category.

## 10. Agent Workflow
Four sub-agents in `.claude/agents/`. Only the orchestrator invokes others.

| Agent | Model | Owns | Invoked when |
|---|---|---|---|
| `@orchestrator` | opus | `config/`, `src/main.py`, `src/utils/`, `src/api/schemas.py`, `docs/`, `README.md`, `Dockerfile`, `docker-compose.yml`, `requirements.txt`, `.env.example` | Project start; integration phase; any cross-agent contract change |
| `@backend` | sonnet | `src/database/`, `src/etl/`, `src/api/routes/`, `src/api/middleware/` | DB schema, ETL pipeline, REST routes, middleware |
| `@ml-engineer` | sonnet | `src/models/`, `src/analytics/`, `notebooks/` | Feature engineering, training, evaluation, clustering, anomaly detection, EDA |
| `@qa-security` | opus | `tests/` only | Code review pass; writing all tests. Reports issues, never edits code outside `tests/`. |

**Phase order:** orchestrator scaffolds → backend + ml-engineer build in parallel against shared schemas → qa-security reviews and writes tests → builders fix → orchestrator integrates `main.py` and `README.md`.

## 11. How to Run (target — orchestrator implements)
```bash
# 1. Bootstrap env
cp .env.example .env                       # then edit secrets

# 2. One-shot via Docker (preferred)
docker compose up --build                  # app on :8000, postgres on :5432

# 3. Bare metal alternative
pip install -r requirements.txt
python -m src.database.init_db             # load CSV/parquet into Postgres
uvicorn src.main:app --reload              # http://localhost:8000

# 4. Tests
pytest -q

# 5. Interactive docs
# http://localhost:8000/docs
```

## 12. Working Rules for Agents
- **Read this file before every task.** It is the single source of truth for contracts.
- **Stay in your lane.** Touch only the files your agent owns. If you need a contract change, escalate to orchestrator.
- **Match dataset columns exactly** in Pydantic schemas — typos are silent bugs.
- **Don't widen scope.** A bug fix doesn't need surrounding cleanup. A schema doesn't need fields the spec didn't list.
- **Don't add backwards-compat shims** — there is no v0; just write v1.
- **No comments explaining WHAT** — names should do that. Comment only when WHY is non-obvious (a constraint, a workaround, a subtle invariant).
