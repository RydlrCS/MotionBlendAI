# Makefile for MotionBlendAI data pipeline
# Two-script runner: sync-test (GCS→BQ) and metrics-export (BQ→ES)

.PHONY: help setup sync-test metrics-export smoke-test clean dbt-run discover

# Environment variables (override with your values)
GCP_PROJECT ?= motionblend-dev
GCS_BUCKET ?= motionblend-mocap
BQ_DATASET ?= RAW_DEV
ES_URL ?= $(ELASTICSEARCH_URL)
ES_INDEX ?= mb_blends_v1

# Python environment
PYTHON := .venv/bin/python
DBT := dbt

help: ## Show this help message
	@echo 'MotionBlendAI Pipeline Commands:'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ''
	@echo 'Environment variables:'
	@echo "  GCP_PROJECT=$(GCP_PROJECT)"
	@echo "  GCS_BUCKET=$(GCS_BUCKET)"
	@echo "  BQ_DATASET=$(BQ_DATASET)"
	@echo "  ES_URL=$(ES_URL)"
	@echo "  ES_INDEX=$(ES_INDEX)"

setup: ## Install dependencies, configure environment
	@echo "📦 Installing dependencies..."
	$(PYTHON) -m pip install -q -r connector/requirements.txt
	$(PYTHON) -m pip install -q -r exporter/requirements.txt
	@echo "✅ Setup complete"

sync-test: ## Run connector: GCS → BigQuery (2 files/category)
	@echo "🔄 Syncing GCS → BigQuery (test mode: 2 files/category)..."
	@echo ""
	@echo "📁 Syncing seed motions..."
	$(PYTHON) connector/run_once.py \
		--bucket $(GCS_BUCKET) \
		--prefix mocap/seed/ \
		--category seed \
		--dataset $(BQ_DATASET) \
		--project $(GCP_PROJECT) \
		--limit 2
	@echo ""
	@echo "📁 Syncing build motions..."
	$(PYTHON) connector/run_once.py \
		--bucket $(GCS_BUCKET) \
		--prefix mocap/build/ \
		--category build \
		--dataset $(BQ_DATASET) \
		--project $(GCP_PROJECT) \
		--limit 2
	@echo ""
	@echo "📁 Syncing blend motions..."
	$(PYTHON) connector/run_once.py \
		--bucket $(GCS_BUCKET) \
		--prefix mocap/blend/ \
		--category blend \
		--dataset $(BQ_DATASET) \
		--project $(GCP_PROJECT) \
		--limit 2
	@echo ""
	@echo "✅ Sync test complete (6 records total)"

metrics-export: ## Run exporter: BigQuery MARTS → Elasticsearch
	@echo "📊 Exporting BigQuery MARTS → Elasticsearch..."
	$(PYTHON) exporter/bigquery_to_elastic.py \
		--bq-project $(GCP_PROJECT) \
		--bq-dataset $(BQ_DATASET) \
		--bq-table mart_blend_snn_metrics \
		--es-url $(ES_URL) \
		--es-index $(ES_INDEX) \
		--batch-size 100
	@echo ""
	@echo "✅ Metrics export complete"

dbt-run: ## Run dbt transformations (staging → marts)
	@echo "🔧 Running dbt transformations..."
	cd dbt_project && \
		$(DBT) deps && \
		$(DBT) debug --profiles-dir . && \
		$(DBT) run --profiles-dir . --target dev && \
		$(DBT) test --profiles-dir .
	@echo "✅ dbt run complete"

discover: ## Show connector schema catalog
	@echo "📋 Connector schema catalog:"
	$(PYTHON) connector/run_once.py --discover-only

smoke-test: sync-test dbt-run metrics-export ## End-to-end test: sync → dbt → export
	@echo ""
	@echo "🎉 Smoke test complete!"
	@echo ""
	@echo "Validation checklist:"
	@echo "  1. Check BigQuery: bq ls $(BQ_DATASET)"
	@echo "  2. Query counts: bq query --nouse_legacy_sql 'SELECT COUNT(*) FROM \`$(GCP_PROJECT).$(BQ_DATASET).mart_blend_snn_metrics\`'"
	@echo "  3. Check Elasticsearch: curl -X GET '$(ES_URL)/$(ES_INDEX)/_count'"

clean: ## Clean state files and logs
	@echo "🧹 Cleaning state and logs..."
	rm -f connector_state.json
	rm -f *.log
	find . -type d -name __pycache__ -exec rm -rf {} +
	@echo "✅ Clean complete"

# Development targets
ui-dev: ## Start UI development server
	@echo "🎨 Starting UI development server..."
	cd ui && npm run dev

compose-up: ## Start Docker Compose services
	@echo "🐳 Starting Docker Compose services..."
	DOCKER_BUILDKIT=1 docker-compose up -d

compose-down: ## Stop Docker services
	docker-compose down

compose-logs: ## Show Docker logs
	docker-compose logs -f

test: ## Run tests
	@echo "🧪 Running tests..."
	$(PYTHON) -m pytest project/tests/ -v

.DEFAULT_GOAL := help
