# MotionBlendAI Architecture - Data & Search Split

## Repository Split Strategy

### 1. motionblend-fivetran-pipeline (This Repo)
- Fivetran SDK connector for GCS motion files
- dbt models for BigQuery transformations
- CI/CD for data pipeline
- Logging & retry infrastructure

### 2. motionblend-elastic-obs-ui (New Repo)
- Elasticsearch indexing
- OBS-style React UI with timeline strips
- Read-only visualization layer
- Batch exporter from BigQuery to Elastic

### 3. fivetran-connector-motionblend (Optional PR)
- Standalone connector for community
- SDK-based motion file ingestion
- Idempotent, chunked processing

## Current Project: blend_snn Terminology

**Note:** Where documentation references `blend_motions`, this project uses `blend_snn`:
- `blend_snn` = Smooth Neural Network blends
- Tables: `RAW.seed_motions`, `RAW.build_motions`, `RAW.blend_snn`
- Metrics: `MARTS.blend_snn_metrics`

## Data Flow (Minimal 2-File Test)

```
GCS (2 seed files) 
  → Fivetran SDK Connector 
  → BigQuery RAW.seed_motions
  → dbt (staging → features → marts)
  → BigQuery MARTS.blend_snn_metrics
  → Batch Exporter (Cloud Run)
  → Elasticsearch mb_blends_v1
  → React OBS UI
```

## Quick Start (Development)

### Prerequisites
```bash
# Install dependencies
make setup

# Configure BigQuery
export GOOGLE_CLOUD_PROJECT=motionblend-dev
export BQ_DATASET=RAW_DEV
```

### Run Minimal Pipeline
```bash
# 1. Sync 2 test files
make sync-test

# 2. Run dbt transformations
make dbt-run

# 3. Export to Elasticsearch
make export-elastic

# 4. Start UI
make ui-dev
```

## Performance Optimizations

### Docker Build Speed
- Multi-stage builds with wheel caching
- BuildKit enabled
- Layer caching to GHCR
- Healthchecks prevent premature starts

### Runtime Performance
- BQ partitioning on `created_at`
- Clustering on `motion_id`
- Export views instead of full tables
- Circuit breaker for failing operations

## Logging & Observability

All services emit JSON logs with:
- `ts`: ISO timestamp
- `level`: DEBUG|INFO|WARN|ERROR
- `service`: connector|dbt|exporter|ui
- `op`: operation name
- `correlation_id`: trace across services
- `duration_ms`: operation timing
- `success`: boolean
- `error_code`: if failed

## Next Steps

See [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) for the 2-week execution roadmap.
