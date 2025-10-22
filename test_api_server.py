#!/usr/bin/env python3
"""
Quick test script for the MotionBlend AI API Server
Tests all endpoints to ensure they work correctly
"""

import requests
import json
import sys
from typing import Dict, Any

# API base URL
API_URL = "http://localhost:8080"

def test_endpoint(name: str, method: str, endpoint: str, data: Dict[str, Any] = None) -> bool:
    """Test a single API endpoint"""
    url = f"{API_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=5)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=5)
        else:
            print(f"❌ {name}: Unsupported method {method}")
            return False
        
        if response.status_code in [200, 201]:
            print(f"✅ {name}: {response.status_code}")
            return True
        else:
            print(f"⚠️  {name}: {response.status_code} - {response.text[:100]}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ {name}: Connection refused (is server running?)")
        return False
    except requests.exceptions.Timeout:
        print(f"❌ {name}: Request timeout")
        return False
    except Exception as e:
        print(f"❌ {name}: {str(e)}")
        return False

def main():
    """Run all API tests"""
    print("🧪 Testing MotionBlend AI API Server")
    print("=" * 60)
    print(f"📍 Base URL: {API_URL}")
    print("=" * 60)
    
    tests = [
        ("Health Check", "GET", "/health"),
        ("Status", "GET", "/status"),
        ("List Motions", "GET", "/motions"),
        ("List Motions (Limited)", "GET", "/motions?limit=5"),
        ("List Motions (Filtered)", "GET", "/motions?category=seed"),
        ("Get Artifacts", "GET", "/api/artifacts"),
        ("Get Artifacts Manifest", "GET", "/api/artifacts/manifest"),
        ("Create Blend", "POST", "/api/blend", {
            "motion1": "walking_forward",
            "motion2": "jump_landing",
            "weight": 0.5
        }),
        ("Vector Search", "POST", "/search/vector", {
            "vector": [0.1] * 384,
            "k": 5
        }),
        ("Root Endpoint", "GET", "/"),
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        name, method, endpoint = test[0], test[1], test[2]
        data = test[3] if len(test) > 3 else None
        
        if test_endpoint(name, method, endpoint, data):
            passed += 1
        else:
            failed += 1
    
    print("=" * 60)
    print(f"📊 Results: {passed} passed, {failed} failed out of {passed + failed} tests")
    print("=" * 60)
    
    if failed == 0:
        print("✅ All tests passed!")
        return 0
    else:
        print(f"⚠️  {failed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
