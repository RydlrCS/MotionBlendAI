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

# Record manifest with input sizes and output sizes
# Write a manifest.json capturing input and output file sizes.
# Use a small Python helper so we don't require jq and we always produce valid JSON.
python3 - <<'PY'
import json, os
root = os.getcwd()
inputs = {}
# list of input files used for the demo (adjust as needed)
input_paths = [
	os.path.join(root, 'build', 'build_motions', 'Air Kicking_fist.glb'),
	os.path.join(root, 'build', 'build_motions', 'Air Kicking_mixamo.glb'),
]
for p in input_paths:
	if os.path.isfile(p):
		inputs[os.path.basename(p)] = os.path.getsize(p)

outputs = []
out_dir = os.path.join(root, 'build', 'demo_artifacts')
if os.path.isdir(out_dir):
	for name in sorted(os.listdir(out_dir)):
		if name == 'manifest.json':
			continue
		path = os.path.join(out_dir, name)
		if os.path.isfile(path):
			outputs.append({"name": name, "size": os.path.getsize(path)})

manifest = {"inputs": inputs, "outputs": outputs}
with open(os.path.join('build', 'demo_artifacts', 'manifest.json'), 'w') as fh:
	json.dump(manifest, fh)

print('Wrote manifest to build/demo_artifacts/manifest.json')
PY
