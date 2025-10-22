# MotionBlendAI Data Pipeline - Execution Guide

## 🎯 Architecture Overview

This pipeline separates data ingestion (Fivetran/BigQuery/dbt) from search UI (Elasticsearch/React) to eliminate infinite loops and improve performance.

**Data Flow:**
```
GCS → Fivetran Connector → BigQuery RAW → dbt → BigQuery MARTS → Batch Exporter → Elasticsearch → React UI
```

## 📦 Components

### 1. Connector (`connector/`)
- **Purpose**: Sync motion files from GCS to BigQuery RAW tables
- **Tables**: `seed_motions`, `build_motions`, `blend_snn`
- **Features**: Idempotent hashing, state management, retry logic, structured logging

### 2. dbt (`dbt_project/`)
- **Purpose**: Transform RAW → STAGE → MARTS
- **Models**: 
  - `staging/`: Clean raw data
  - `marts/`: Compute blend quality metrics (FID, coverage, velocity, acceleration)

### 3. Exporter (`exporter/`)
- **Purpose**: Batch export BigQuery MARTS → Elasticsearch
- **Features**: Bulk indexing, circuit breaker, correlation IDs

### 4. Docker (`docker/`)
- **Multi-stage builds**: Separate build wheelhouse from runtime
- **Healthchecks**: Prevent flapping dependencies
- **Split services**: connector, dbt-runner, exporter, ui, api

## 🚀 Quick Start

### Prerequisites

```bash
export GCP_PROJECT="your-gcp-project"
export GCS_BUCKET="motionblend-mocap"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
export ELASTICSEARCH_URL="https://your-cluster.es.cloud:9243"
export ELASTICSEARCH_API_KEY="your-api-key"
```

### Step 1: Setup

```bash
# Install dependencies
make setup

# Verify connector schema
make discover
```

Expected output:
```
📋 Schema Catalog:

  seed_motions:
    - id
    - file_uri
    - skeleton_id
    - frames
    - fps
    - joints_count
    - created_at
    - updated_at
    
  build_motions: [...]
  blend_snn: [...]
```

### Step 2: Upload Test Files to GCS

```bash
# Upload 2 files per category (6 total)
gsutil cp tests/seed/*.bvh gs://${GCS_BUCKET}/mocap/seed/
gsutil cp tests/build/*.bvh gs://${GCS_BUCKET}/mocap/build/
gsutil cp tests/blend/*.bvh gs://${GCS_BUCKET}/mocap/blend/

# Verify
gsutil ls gs://${GCS_BUCKET}/mocap/**/*.bvh
```

### Step 3: Create BigQuery Dataset

```bash
bq mk --project_id=${GCP_PROJECT} RAW_DEV
```

### Step 4: Run First Sync (GCS → BigQuery)

```bash
# Sync all categories (2 files each = 6 total)
make sync-test
```

Expected output:
```
✅ Successfully synced 2 records
   Category: seed
   Dataset: RAW_DEV
   Table: seed_motions
   
[... similar for build and blend ...]
```

Verify in BigQuery:
```bash
bq query --nouse_legacy_sql \
  "SELECT 'seed' t, COUNT(*) c FROM \`${GCP_PROJECT}.RAW_DEV.seed_motions\`
   UNION ALL
   SELECT 'build', COUNT(*) FROM \`${GCP_PROJECT}.RAW_DEV.build_motions\`
   UNION ALL
   SELECT 'blend', COUNT(*) FROM \`${GCP_PROJECT}.RAW_DEV.blend_snn\`"
```

Expected:
```
+-------+---+
|   t   | c |
+-------+---+
| seed  | 2 |
| build | 2 |
| blend | 2 |
+-------+---+
```

### Step 5: Configure dbt

```bash
# Copy profiles to home directory
cp dbt_project/profiles.yml ~/.dbt/profiles.yml

# Edit with your GCP project
vim ~/.dbt/profiles.yml
# Set: project: "your-gcp-project"
# Set: keyfile: "/path/to/service-account.json"

# Test connection
cd dbt_project && dbt debug --profiles-dir ~/.dbt
```

### Step 6: Run dbt Transformations

```bash
make dbt-run
```

Expected output:
```
🔧 Running dbt transformations...
✅ dbt run complete
```

Verify marts:
```bash
bq query --nouse_legacy_sql \
  "SELECT * FROM \`${GCP_PROJECT}.RAW_DEV.mart_blend_snn_metrics\` LIMIT 5"
```

### Step 7: Export to Elasticsearch

```bash
make metrics-export
```

Expected output:
```
✅ Successfully exported 2 records
   From: your-project.RAW_DEV.mart_blend_snn_metrics
   To: mb_blends_v1
```

Verify index:
```bash
curl -X GET "${ELASTICSEARCH_URL}/mb_blends_v1/_count" \
  -H "Authorization: ApiKey ${ELASTICSEARCH_API_KEY}"
```

### Step 8: Run Full Smoke Test

```bash
# End-to-end: sync → dbt → export
make smoke-test
```

## 🐳 Docker Usage

### Build Images

```bash
# Build all services with BuildKit
DOCKER_BUILDKIT=1 docker-compose -f docker-compose.optimized.yml build
```

### Run Pipeline

```bash
# Set environment variables in .env file
cat > .env <<EOF
GCP_PROJECT=your-gcp-project
GCS_BUCKET=motionblend-mocap
BQ_DATASET=RAW_DEV
GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json
ELASTICSEARCH_URL=https://your-cluster.es.cloud:9243
ELASTICSEARCH_API_KEY=your-api-key
EOF

# Run connector
docker-compose -f docker-compose.optimized.yml up connector

# Run dbt
docker-compose -f docker-compose.optimized.yml up dbt-runner

# Run exporter
docker-compose -f docker-compose.optimized.yml up exporter

# Or run all in sequence
docker-compose -f docker-compose.optimized.yml up
```

## 🔍 Observability

### Logs

All components use structured JSON logging with:
- `correlation_id`: Track requests across services
- `duration_ms`: Measure operation timing
- `event`: High-level operation name
- `context`: Additional metadata

Example log entry:
```json
{
  "timestamp": "2025-10-22T10:30:45.123Z",
  "level": "info",
  "event": "sync_category",
  "correlation_id": "abc123",
  "category": "seed",
  "records_synced": 2,
  "duration_ms": 3421
}
```

### Retry Behavior

- **Base delay**: 0.5s
- **Backoff factor**: 2× (exponential)
- **Max delay**: 30s
- **Max attempts**: 7
- **Jitter**: 10%

Circuit breaker trips at 50% failure rate, recovers after 120s.

### State Management

State is persisted in `connector_state.json`:
```json
{
  "cursors": {
    "seed": {
      "value": "2025-10-22T10:30:00Z",
      "updated_at": "2025-10-22T10:30:45Z"
    }
  },
  "sync_history": {
    "seed": {
      "total_records": 2,
      "sync_count": 1,
      "last_sync": "2025-10-22T10:30:45Z"
    }
  }
}
```

## ⚙️ Configuration

### Connector (`connector/config.yaml`)

```yaml
source:
  bucket: motionblend-mocap
  prefixes:
    - mocap/seed/
    - mocap/build/
    - mocap/blend/

destination:
  project: your-gcp-project
  dataset: RAW_DEV
  
batch:
  size: 25
  max_retries: 5
  timeout_seconds: 300
```

### dbt (`dbt_project/dbt_project.yml`)

```yaml
models:
  motionblend:
    staging:
      +materialized: view
      +schema: staging
    marts:
      +materialized: table
      +schema: marts
      +partition_by:
        field: created_at
        data_type: timestamp
```

## 🧪 Testing

### Unit Tests

```bash
make test
```

### Integration Test (requires GCS access)

```bash
# Test with real GCS bucket
python connector/run_once.py \
  --bucket ${GCS_BUCKET} \
  --prefix mocap/seed/ \
  --limit 1
```

### SQL Validation Queries

```sql
-- Check data quality
SELECT 
  quality_category,
  COUNT(*) as count,
  AVG(quality_score) as avg_score,
  AVG(fid) as avg_fid
FROM `your-project.RAW_DEV.mart_blend_snn_metrics`
GROUP BY quality_category;

-- Verify all blends have source motions
SELECT 
  b.blend_id,
  b.left_motion_id,
  b.right_motion_id,
  sm1.id as left_exists,
  sm2.id as right_exists
FROM `your-project.RAW_DEV.mart_blend_snn_metrics` b
LEFT JOIN `your-project.RAW_DEV.seed_motions` sm1 ON b.left_motion_id = sm1.id
LEFT JOIN `your-project.RAW_DEV.seed_motions` sm2 ON b.right_motion_id = sm2.id
WHERE sm1.id IS NULL OR sm2.id IS NULL;
```

## 📊 Performance Targets

- **Flask startup**: <1 second (no Elasticsearch in critical path)
- **Docker build**: <2 minutes (multi-stage + cache)
- **Connector (2 files)**: <5 minutes
- **dbt batch (6 files)**: <10 minutes
- **Elasticsearch export**: <3 minutes
- **No infinite loops**: Circuit breakers prevent runaway processes

## 🚨 Troubleshooting

### Connector fails with GCS permission error

```bash
# Verify service account has Storage Object Viewer role
gcloud projects get-iam-policy ${GCP_PROJECT} \
  --flatten="bindings[].members" \
  --filter="bindings.role:roles/storage.objectViewer"
```

### dbt connection fails

```bash
# Test BigQuery access
bq ls --project_id=${GCP_PROJECT}

# Verify profiles.yml
cd dbt_project && dbt debug --profiles-dir ~/.dbt
```

### Elasticsearch indexing slow

Increase batch size:
```bash
python exporter/bigquery_to_elastic.py \
  --batch-size 500  # default: 100
```

### Docker build slow

Enable BuildKit and use cache:
```bash
DOCKER_BUILDKIT=1 docker-compose -f docker-compose.optimized.yml build \
  --build-arg BUILDKIT_INLINE_CACHE=1
```

## 🎯 Next Steps

1. **Scale up**: Increase `--limit` from 2 to process all files
2. **Schedule**: Set up Cloud Scheduler to run connector daily
3. **Monitor**: Add Datadog/Prometheus for metrics
4. **Alerting**: Configure PagerDuty for circuit breaker trips
5. **CI/CD**: GitHub Actions for lint, test, build, deploy

## 📚 Resources

- [Architecture doc](ARCHITECTURE.md)
- [Connector modules](connector/)
- [dbt models](dbt_project/models/)
- [Docker configs](docker/)
- [Makefile targets](Makefile)
