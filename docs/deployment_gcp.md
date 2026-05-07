# Deploying to Google Cloud (Cloud Run + Cloud SQL)

This is the MVP-grade GCP deployment path. Cloud Run for the app
(serverless, scales to zero, pay per request), Cloud SQL Postgres for
the data, Secret Manager for credentials, Artifact Registry for the
image, Cloud Build for the pipeline.

> **Important — do NOT suppress stderr on `gcloud sql instances create`.**
> If the create fails (quota, billing-pending, deprecated tier), the error
> message is the only signal. Without it the script keeps running and
> every subsequent SQL Admin API call returns 403 (because the instance
> doesn't exist AND you can't enumerate non-existent instances). This
> error chain is hard to diagnose and wasted ~30 min of debug time on
> a real deployment.

## Architecture

```
            Public HTTPS (managed cert)
                       │
                       ▼
               ┌───────────────┐
               │   Cloud Run   │  fmcg-app  (1 vCPU / 1 GiB)
               │  Python 3.12  │  scales 0..4
               └───────┬───────┘
                       │ Unix socket
                       │ /cloudsql/PROJECT:REGION:INSTANCE
                       ▼
               ┌───────────────┐
               │   Cloud SQL   │  postgres-16-alpine equivalent
               │   Postgres    │  db-f1-micro for MVP
               └───────────────┘
```

Trained model artifacts (~5 MB) are baked into the image. Secrets
(`DB_PASSWORD`, `API_KEY`) live in Secret Manager and are injected as
env vars at deploy time.

---

## One-time setup

Replace the placeholders. Run from the repo root.

```bash
# 1. Variables
PROJECT_ID="<your-gcp-project>"
REGION="us-central1"
SQL_INSTANCE="fmcg-pg"
SQL_TIER="db-f1-micro"
DB_NAME="fmcg_intelligence"
DB_USER="fmcg_user"
SERVICE="fmcg-app"
REPO="fmcg"

# Strong dev password (also rotated into Secret Manager below)
DB_PASSWORD="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"

gcloud config set project "$PROJECT_ID"

# 2. Enable APIs
gcloud services enable \
  run.googleapis.com \
  sqladmin.googleapis.com \
  secretmanager.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com

# 3. Artifact Registry repo (Docker)
gcloud artifacts repositories create "$REPO" \
  --repository-format=docker \
  --location="$REGION" \
  --description="FMCG Demand Forecasting images"

IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/app"

# 4. Cloud SQL instance + database + user
gcloud sql instances create "$SQL_INSTANCE" \
  --database-version=POSTGRES_16 \
  --tier="$SQL_TIER" \
  --region="$REGION" \
  --storage-size=10GB \
  --storage-auto-increase

gcloud sql databases create "$DB_NAME" --instance="$SQL_INSTANCE"
gcloud sql users create "$DB_USER" --instance="$SQL_INSTANCE" --password="$DB_PASSWORD"

CLOUD_SQL_CONN="${PROJECT_ID}:${REGION}:${SQL_INSTANCE}"

# 5. Secret Manager
printf "%s" "$DB_PASSWORD" | gcloud secrets create db-password --data-file=-
printf "%s" "$API_KEY"     | gcloud secrets create api-key      --data-file=-

# 6. Grant the Cloud Run runtime service account access to secrets and Cloud SQL
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')
RUNTIME_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

gcloud secrets add-iam-policy-binding db-password \
  --member="serviceAccount:${RUNTIME_SA}" \
  --role="roles/secretmanager.secretAccessor"
gcloud secrets add-iam-policy-binding api-key \
  --member="serviceAccount:${RUNTIME_SA}" \
  --role="roles/secretmanager.secretAccessor"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${RUNTIME_SA}" \
  --role="roles/cloudsql.client"

# 7. Echo back values you'll need
echo "IMAGE=$IMAGE"
echo "CLOUD_SQL_CONN=$CLOUD_SQL_CONN"
echo "DB_USER=$DB_USER  DB_NAME=$DB_NAME"
echo "API_KEY=$API_KEY    # save this somewhere secure"
```

---

## Initial schema + data load

The `init_db.py` script needs to run **once** against the Cloud SQL
instance to create the schema and load the 222k rows. Two options:

### Option A — Local proxy (simplest)

```bash
# 1. Install the proxy if needed:
#    https://cloud.google.com/sql/docs/postgres/sql-proxy

# 2. Start the proxy (background)
./cloud-sql-proxy "$CLOUD_SQL_CONN" --port=5433 &

# 3. Run init_db with the local proxy as DB_HOST
DB_HOST=127.0.0.1 \
DB_PORT=5433 \
DB_NAME="$DB_NAME" \
DB_USER="$DB_USER" \
DB_PASSWORD="$DB_PASSWORD" \
RAW_DATA_DIR=data/raw \
python -m src.database.init_db

# 4. Stop the proxy
kill %1
```

### Option B — One-shot Cloud Run Job (no local proxy)

```bash
gcloud run jobs create fmcg-init-db \
  --image="${IMAGE}:latest" \
  --region="$REGION" \
  --add-cloudsql-instances="$CLOUD_SQL_CONN" \
  --set-env-vars="DB_HOST=/cloudsql/${CLOUD_SQL_CONN},DB_NAME=${DB_NAME},DB_USER=${DB_USER}" \
  --set-secrets="DB_PASSWORD=db-password:latest" \
  --command=python --args="-m,src.database.init_db" \
  --memory=1Gi --task-timeout=300

gcloud run jobs execute fmcg-init-db --region="$REGION" --wait
```

(Run the build first — see next section — so the image exists.)

---

## Build + deploy

The repo's `cloudbuild.yaml` does build → push → deploy in one command.

```bash
gcloud builds submit \
  --config=cloudbuild.yaml \
  --substitutions=_REGION="$REGION",_SERVICE="$SERVICE",_CLOUD_SQL_INSTANCE="$CLOUD_SQL_CONN",_DB_NAME="$DB_NAME",_DB_USER="$DB_USER",_IMAGE="$IMAGE"
```

Expected wall time: ~3–5 minutes (build), ~30 seconds (deploy).

After deploy succeeds, gcloud prints the public URL:

```
Service URL: https://fmcg-app-<hash>-uc.a.run.app
```

---

## Smoke tests against the live deployment

```bash
URL="<paste from the deploy output>"

# 1. Health (no auth — should return 200)
curl -sf "$URL/api/v1/health" | python -m json.tool

# 2. Forecast (auth required)
curl -sf -H "X-API-Key: $API_KEY" \
  "$URL/api/v1/forecast/MI-006?channel=Retail&region=PL-Central&horizon_weeks=4" \
  | python -m json.tool

# 3. Sales summary
curl -sf -H "X-API-Key: $API_KEY" \
  "$URL/api/v1/analytics/sales-summary?start_date=2024-01-01&end_date=2024-12-31" \
  | python -m json.tool

# 4. Auth gates
curl -s -o /dev/null -w "%{http_code}\n" "$URL/api/v1/forecast/MI-006"   # 401
curl -s -o /dev/null -w "%{http_code}\n" -H "X-API-Key: wrong" \
  "$URL/api/v1/forecast/MI-006"                                          # 401
```

OpenAPI docs: `${URL}/docs`.

---

## Cost (rough MVP estimate, us-central1)

| Component                | Idle               | Active (10k req/day) |
|--------------------------|--------------------|----------------------|
| Cloud Run (1 vCPU, 1 GiB) | $0 (scales to 0)  | ~$3–8/month          |
| Cloud SQL (db-f1-micro)   | ~$8/month         | ~$8–15/month         |
| Secret Manager            | ~$0.06/secret/mo  | ~$0.12/month         |
| Artifact Registry         | ~$0.10/GB/mo      | <$1/month            |
| Cloud Build               | $0 (free tier)    | $0 in free tier      |
| **Total MVP**             | **~$10/month**    | **~$15–25/month**    |

Stop-the-bill checklist:
```bash
gcloud run services delete "$SERVICE" --region="$REGION"
gcloud sql instances delete "$SQL_INSTANCE"
gcloud artifacts repositories delete "$REPO" --location="$REGION"
gcloud secrets delete db-password
gcloud secrets delete api-key
```

---

## MVP-to-production gaps (what to harden later)

1. **Rate limiter is per-instance.** Cloud Run scales horizontally; the
   in-memory sliding-window limiter only sees requests for one instance.
   With `max-instances=4` and `concurrency=20`, the effective per-key
   ceiling is `4 × 100 = 400 req/min`. For real production, move to
   Memorystore (Redis) or use Cloud Armor with rate-based rules.

2. **Models in image.** Re-baking the image on every retrain is fine at
   ~5 MB and weekly cadence. Past that, store artifacts in Cloud Storage
   and pull at startup in `src/main.py`'s lifespan handler.

3. **No automated retraining schedule.** MVP runs `scripts/train_all_models.py`
   manually. Production wants a scheduled Cloud Run Job or Cloud Composer
   DAG triggering on a weekly cadence.

4. **API key auth only.** Single shared key. For per-tenant access,
   migrate to Identity-Aware Proxy or issue per-customer keys with
   IAM.

5. **No request tracing / metrics dashboard.** Cloud Run sends logs to
   Cloud Logging automatically; for SLO-grade observability, add OpenTelemetry
   instrumentation and a Cloud Monitoring dashboard.

6. **One Cloud SQL instance, one zone.** db-f1-micro has no HA. For
   production, switch to a regional-HA tier and enable point-in-time
   recovery.
