#!/usr/bin/env bash
set -euo pipefail

# Ensure local project dir is importable
export PYTHONPATH="/app:${PYTHONPATH:-}"

# Run vertex_pipeline tests
pytest -q project/vertex_pipeline -q

echo "All tests passed"
