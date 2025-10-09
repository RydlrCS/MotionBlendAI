#!/usr/bin/env python3
"""
Test script for Elasticsearch API endpoints
"""

import requests
import json
import time

BASE_URL = "http://localhost:5002"

def test_health():
    """Test health endpoint"""
    print("🏥 Testing Health Endpoint")
    print("=" * 40)
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_search():
    """Test vector search endpoint"""
    print("\n🔍 Testing Vector Search")
    print("=" * 40)
    
    search_data = {
        "vector": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "size": 3
    }
    
    try:
        response = requests.post(f"{BASE_URL}/search", 
                               json=search_data, 
                               timeout=5)
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Found {len(result.get('results', []))} motions")
        
        for i, motion in enumerate(result.get('results', [])[:3], 1):
            print(f"{i}. {motion['name']} (score: {motion.get('score', 'N/A')})")
        
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_text_search():
    """Test text search endpoint"""
    print("\n📝 Testing Text Search")
    print("=" * 40)
    
    search_data = {
        "query": "dance",
        "size": 3
    }
    
    try:
        response = requests.post(f"{BASE_URL}/search/text", 
                               json=search_data, 
                               timeout=5)
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Found {len(result.get('results', []))} motions")
        
        for i, motion in enumerate(result.get('results', [])[:3], 1):
            print(f"{i}. {motion['name']}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_mock_data():
    """Test mock data endpoint"""
    print("\n🎭 Testing Mock Data")
    print("=" * 40)
    
    try:
        response = requests.get(f"{BASE_URL}/motions", timeout=5)
        print(f"Status: {response.status_code}")
        motions = response.json()
        print(f"Total mock motions: {len(motions)}")
        
        for i, motion in enumerate(motions[:3], 1):
            print(f"{i}. {motion['name']} - {motion['metadata']['category']}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 MotionBlend AI Elasticsearch API Test")
    print("=" * 50)
    
    # Wait for server startup
    print("⏳ Waiting for server startup...")
    time.sleep(2)
    
    tests = [
        ("Health Check", test_health),
        ("Vector Search", test_search), 
        ("Text Search", test_text_search),
        ("Mock Data", test_mock_data)
    ]
    
    results = []
    for test_name, test_func in tests:
        success = test_func()
        results.append((test_name, success))
    
    print("\n📊 Test Results Summary")
    print("=" * 50)
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if success:
            passed += 1
    
    print(f"\n🎯 {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 All tests passed! Elasticsearch API is working correctly.")
        print("\n📝 Next steps:")
        print("1. Update frontend client to use port 5002")
        print("2. Test bulk indexing with real motion data")
        print("3. Implement advanced search features")
    else:
        print("\n⚠️  Some tests failed. Check server logs for details.")