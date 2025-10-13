#!/usr/bin/env python3
"""
Simple API test script to verify MotionBlendAI endpoints
"""
import requests
import json

API_BASE = "http://localhost:5000"

def test_endpoint(endpoint, method="GET", data=None):
    """Test an API endpoint and return result"""
    try:
        url = f"{API_BASE}{endpoint}"
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        
        print(f"✅ {method} {endpoint}: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list):
                print(f"   📊 Returned {len(result)} items")
            elif isinstance(result, dict):
                print(f"   📊 Keys: {list(result.keys())}")
            return result
        else:
            print(f"   ❌ Error: {response.text}")
            return None
    except Exception as e:
        print(f"❌ {method} {endpoint}: Failed - {e}")
        return None

if __name__ == "__main__":
    print("🚀 Testing MotionBlendAI API endpoints...")
    print("=" * 50)
    
    # Test health endpoint
    test_endpoint("/health")
    
    # Test motions endpoint
    motions = test_endpoint("/motions")
    
    # Test search endpoints with sample data
    print("\n🔍 Testing search endpoints...")
    
    # Vector search test
    test_data = {
        "vector": [0.1, 0.2, 0.3, 0.4, 0.5],
        "k": 3
    }
    test_endpoint("/search/vector", "POST", test_data)
    
    # Semantic search test
    test_data = {
        "query": "walking motion",
        "limit": 5
    }
    test_endpoint("/search/semantic", "POST", test_data)
    
    print("\n✨ API testing complete!")