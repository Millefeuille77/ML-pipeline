#!/usr/bin/env bash
# =============================================================================
# scheduler.gcloud.sh — Cloud Scheduler job definitions for FMCG platform.
#
# IMPORTANT: ALL SCHEDULES ARE CREATED PAUSED.
# The `--paused` flag is intentional. The user wants the schedule defined in
# infrastructure as code but NOT firing automatically. A human must explicitly
# run `gcloud scheduler jobs resume <JOB_NAME>` to activate each schedule.
#
# Schedules defined:
#   fmcg-retrain-weekly  — Mondays 02:00 UTC  — runs fmcg-retrain Cloud Run Job
#   fmcg-score-daily     — Daily    03:00 UTC  — runs fmcg-score   Cloud Run Job
#
# Prerequisites:
#   1. Cloud Run Jobs deployed:
#        gcloud run jobs replace scripts/cloud_jobs/retrain_job.yaml --region=$REGION
#        gcloud run jobs replace scripts/cloud_jobs/score_job.yaml   --region=$REGION
#   2. Service account $SCHEDULER_SA must have role roles/run.invoker on both jobs.
#   3. Set the four variables below before running this script.
#
# Usage:
#   export PROJECT_ID=my-project
#   export REGION=us-central1
#   export SCHEDULER_SA=scheduler-sa@my-project.iam.gserviceaccount.com
#   bash scripts/cloud_jobs/scheduler.gcloud.sh
#
# To resume when ready:
#   gcloud scheduler jobs resume fmcg-retrain-weekly --location=$REGION
#   gcloud scheduler jobs resume fmcg-score-daily     --location=$REGION
# =============================================================================

set -euo pipefail

: "${PROJECT_ID:?Set PROJECT_ID}"
: "${REGION:?Set REGION}"
: "${SCHEDULER_SA:?Set SCHEDULER_SA}"

RETRAIN_JOB_URI="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_ID}/jobs/fmcg-retrain:run"
SCORE_JOB_URI="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_ID}/jobs/fmcg-score:run"

# -----------------------------------------------------------------------------
# Create the weekly retraining scheduler job (PAUSED at creation).
# Schedule: every Monday at 02:00 UTC.
# PAUSED: the schedule is defined but will NOT fire until manually resumed.
# -----------------------------------------------------------------------------
gcloud scheduler jobs create http fmcg-retrain-weekly \
  --location="${REGION}" \
  --schedule="0 2 * * 1" \
  --time-zone="UTC" \
  --uri="${RETRAIN_JOB_URI}" \
  --message-body="{}" \
  --oauth-service-account-email="${SCHEDULER_SA}" \
  --paused \
  --description="Weekly retraining pipeline (PAUSED — resume manually to activate)"

echo "Created fmcg-retrain-weekly scheduler job [PAUSED]"

# -----------------------------------------------------------------------------
# Create the daily scoring scheduler job (PAUSED at creation).
# Schedule: every day at 03:00 UTC.
# PAUSED: the schedule is defined but will NOT fire until manually resumed.
# -----------------------------------------------------------------------------
gcloud scheduler jobs create http fmcg-score-daily \
  --location="${REGION}" \
  --schedule="0 3 * * *" \
  --time-zone="UTC" \
  --uri="${SCORE_JOB_URI}" \
  --message-body="{}" \
  --oauth-service-account-email="${SCHEDULER_SA}" \
  --paused \
  --description="Daily prediction scoring job (PAUSED — resume manually to activate)"

echo "Created fmcg-score-daily scheduler job [PAUSED]"

# -----------------------------------------------------------------------------
# Explicit pause commands for documentation purposes.
# These are no-ops immediately after creation (already paused), but serve as
# runbook entries if the jobs are accidentally resumed and need to be paused.
# -----------------------------------------------------------------------------
gcloud scheduler jobs pause fmcg-retrain-weekly --location="${REGION}"
gcloud scheduler jobs pause fmcg-score-daily     --location="${REGION}"

echo ""
echo "Both scheduler jobs are PAUSED."
echo "To activate when ready:"
echo "  gcloud scheduler jobs resume fmcg-retrain-weekly --location=${REGION}"
echo "  gcloud scheduler jobs resume fmcg-score-daily     --location=${REGION}"
