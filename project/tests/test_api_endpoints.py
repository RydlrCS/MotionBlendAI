#!/usr/bin/env python3
"""
Test script for the updated Flask API endpoints with semantic search capabilities.
"""

import json
import time
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

def test_api_endpoints():
    """Test all the new and updated API endpoints."""
    print("🧪 Testing Updated Flask API Endpoints")
    print("=" * 50)
    
    # Import the Flask app
    try:
        import sys
        import os
        sys.path.append('/Users/ted/blenderkit_data/MotionBlendAI-1/project/elastic_search')
        from app import app, MOCK_MOTIONS, es_available
        print(f"✅ Flask app imported successfully")
        print(f"📊 Mock motions available: {len(MOCK_MOTIONS)}")
        print(f"🔌 Elasticsearch available: {es_available}")
    except ImportError as e:
        print(f"❌ Failed to import Flask app: {e}")
        return
    
    # Create test client
    with app.test_client() as client:
        print("\n🏥 Testing Health Endpoint")
        print("-" * 30)
        
        response = client.get('/health')
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            health_data = response.get_json()
            print(f"✅ Health check passed")
            print(f"  Status: {health_data.get('status')}")
            print(f"  Elasticsearch: {health_data.get('elasticsearch_available')}")
            print(f"  Mock motions: {health_data.get('mock_motions_count')}")
            print(f"  Index name: {health_data.get('index_name')}")
        else:
            print(f"❌ Health check failed")
        
        print("\n🔍 Testing Vector Search Endpoint")
        print("-" * 35)
        
        test_vector = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        search_payload = {
            "vector": test_vector,
            "k": 3
        }
        
        response = client.post('/search', 
                             data=json.dumps(search_payload),
                             content_type='application/json')
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            results = response.get_json()
            print(f"✅ Vector search successful")
            print(f"  Results: {len(results)}")
            for i, result in enumerate(results[:2]):
                print(f"  {i+1}. {result.get('name', 'Unknown')} (score: {result.get('similarity_score', 0):.3f})")
        else:
            print(f"❌ Vector search failed: {response.get_json()}")
        
        print("\n📝 Testing Text Search Endpoint")
        print("-" * 32)
        
        text_payload = {
            "query": "walking movement",
            "k": 3
        }
        
        response = client.post('/search/text',
                             data=json.dumps(text_payload),
                             content_type='application/json')
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            results = response.get_json()
            print(f"✅ Text search successful")
            print(f"  Results: {len(results)}")
            for i, result in enumerate(results[:2]):
                print(f"  {i+1}. {result.get('name', 'Unknown')} (score: {result.get('similarity_score', 0):.3f})")
        else:
            print(f"❌ Text search failed: {response.get_json()}")
        
        print("\n🧠 Testing Semantic Search Endpoint")
        print("-" * 36)
        
        semantic_payload = {
            "query": "athletic performance with dynamic movement",
            "k": 3
        }
        
        response = client.post('/search/semantic',
                             data=json.dumps(semantic_payload),
                             content_type='application/json')
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            results = response.get_json()
            print(f"✅ Semantic search successful")
            print(f"  Query: {results.get('query')}")
            print(f"  Results: {results.get('total', 0)}")
            for i, result in enumerate(results.get('results', [])[:2]):
                print(f"  {i+1}. {result.get('name', 'Unknown')} (score: {result.get('semantic_score', 0):.3f})")
        else:
            print(f"❌ Semantic search failed: {response.get_json()}")
        
        print("\n🔀 Testing Hybrid Search Endpoint")
        print("-" * 33)
        
        hybrid_payload = {
            "vector": test_vector,
            "query": "dynamic motion",
            "k": 3,
            "vector_weight": 0.6
        }
        
        response = client.post('/search/hybrid',
                             data=json.dumps(hybrid_payload),
                             content_type='application/json')
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            results = response.get_json()
            print(f"✅ Hybrid search successful")
            print(f"  Vector weight: {results.get('weights', {}).get('vector', 0)}")
            print(f"  Text weight: {results.get('weights', {}).get('text', 0)}")
            print(f"  Results: {results.get('total', 0)}")
            for i, result in enumerate(results.get('results', [])[:2]):
                print(f"  {i+1}. {result.get('name', 'Unknown')} (score: {result.get('hybrid_score', 0):.3f})")
        else:
            print(f"❌ Hybrid search failed: {response.get_json()}")
        
        print("\n📥 Testing Bulk Index Endpoint")
        print("-" * 31)
        
        test_documents = [
            {
                "id": "test_motion_1",
                "name": "Test Athletic Jump",
                "description": "High-energy jumping motion with athletic characteristics",
                "motion_vector": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
                "metadata": {
                    "category": "athletic",
                    "tags": ["jumping", "dynamic", "sports"],
                    "duration": 2.5
                }
            },
            {
                "id": "test_motion_2", 
                "name": "Gentle Yoga Flow",
                "description": "Calm and controlled yoga sequence with flowing movements",
                "motion_vector": [0.2, 0.3, 0.1, 0.4, 0.2, 0.3, 0.1, 0.5],
                "metadata": {
                    "category": "wellness",
                    "tags": ["yoga", "meditation", "gentle"],
                    "duration": 8.0
                }
            }
        ]
        
        bulk_payload = {
            "documents": test_documents
        }
        
        response = client.post('/index/bulk',
                             data=json.dumps(bulk_payload),
                             content_type='application/json')
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            results = response.get_json()
            print(f"✅ Bulk indexing successful")
            print(f"  Indexed: {results.get('indexed', 0)}")
            print(f"  Errors: {results.get('errors', 0)}")
            print(f"  Semantic processing: {results.get('semantic_processing', False)}")
        else:
            print(f"❌ Bulk indexing failed: {response.get_json()}")
        
        print("\n📋 Testing Motions List Endpoint")
        print("-" * 34)
        
        response = client.get('/motions')
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            motions = response.get_json()
            print(f"✅ Motions list successful")
            print(f"  Total motions: {len(motions)}")
            for i, motion in enumerate(motions[:3]):
                print(f"  {i+1}. {motion.get('name', 'Unknown')} ({motion.get('metadata', {}).get('category', 'unknown')})")
        else:
            print(f"❌ Motions list failed")
    
    print("\n🎉 API Endpoint Testing Complete!")
    print("=" * 50)

if __name__ == '__main__':
    test_api_endpoints()