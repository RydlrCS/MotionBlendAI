#!/bin/bash
# Quick start script for local API testing

echo "🚀 Starting MotionBlend AI API Server"
echo "======================================"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q flask flask-cors google-cloud-storage google-cloud-bigquery elasticsearch requests

# Set environment variables
export GCS_BUCKET="${GCS_BUCKET:-motionblend-mocap}"
export BQ_PROJECT="${BQ_PROJECT:-motionblend-ai}"
export BQ_DATASET="${BQ_DATASET:-RAW_DEV}"
export ELASTICSEARCH_URL="${ELASTICSEARCH_URL:-http://localhost:9200}"

echo ""
echo "Environment:"
echo "  GCS_BUCKET: $GCS_BUCKET"
echo "  BQ_PROJECT: $BQ_PROJECT"
echo "  BQ_DATASET: $BQ_DATASET"
echo "  ELASTICSEARCH_URL: $ELASTICSEARCH_URL"
echo ""

# Start server
echo "Starting server on http://localhost:8080"
echo "Press Ctrl+C to stop"
echo "======================================"
python api_server.py
