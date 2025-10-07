#!/usr/bin/env bash
# Simple demo runner: runs SNN blend then SPADE inference and collects outputs
set -euo pipefail
ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
echo "Running demo from $ROOT_DIR"

cd "$ROOT_DIR"

# Ensure output directories
mkdir -p build/blend_snn
mkdir -p build/demo_artifacts

echo "Running SNN blend on two build motions..."
python3 project/ganimator/motion_extractor.py blend_snn "Air Kicking_fist.glb" "Air Kicking_mixamo.glb" 0.6

echo "Running SPADE infer to generate sample blended tensor (prints to stdout)..."
python3 project/ganimator/infer.py > build/demo_artifacts/spade_infer.log 2>&1 || true

echo "Collecting artifacts..."
cp -v build/blend_snn/*.npy build/demo_artifacts/ 2>/dev/null || true
ls -la build/demo_artifacts

echo "Demo complete. Artifacts are in build/demo_artifacts and blend_snn."
