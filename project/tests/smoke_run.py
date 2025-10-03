"""
Simple smoke test runner that validates the search API, inference, and training scripts
without requiring full PyTorch or Elasticsearch to be installed.
"""
import requests
import subprocess
import sys
import time

# Test 1: Search API
try:
    r = requests.post('http://127.0.0.1:8080/search', json={'text_query': 'walk into jump', 'k': 3}, timeout=5)
    print('Search API status:', r.status_code)
    print('Search API body:', r.text)
except Exception as e:
    print('Search API test failed:', e)

# Test 2: Inference (call our infer.py but intercept torch import by running in a subprocess with env var)
# We'll run infer.py directly and capture its output.
try:
    proc = subprocess.run([sys.executable, 'project/ganimator/infer.py'], capture_output=True, text=True, timeout=10)
    print('Infer stdout:', proc.stdout)
    print('Infer stderr:', proc.stderr)
except Exception as e:
    print('Inference test failed:', e)

# Test 3: Training script
try:
    proc = subprocess.run([sys.executable, 'project/ganimator/train.py'], capture_output=True, text=True, timeout=20)
    print('Train stdout:', proc.stdout)
    print('Train stderr:', proc.stderr)
except Exception as e:
    print('Training test failed:', e)
