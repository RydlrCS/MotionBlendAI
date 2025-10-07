#!/usr/bin/env bash
set -euo pipefail

# Run tests inside container
pytest -q project/vertex_pipeline/test_trigger_pipeline_function.py -q || exit 1
pytest -q project/vertex_pipeline/test_gsutil_console.py -q || exit 1

echo "All tests passed"
