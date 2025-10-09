#!/usr/bin/env python3
"""
MotionBlend AI Elasticsearch API - Optimized Version
Fast startup with lazy Elasticsearch initialization
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
ES_INDEX_NAME = "motion-blend"
ES_CLOUD_URL = "https://my-elasticsearch-project-ba986d.es.us-central1.gcp.elastic.cloud:443"
ES_API_KEY = "S21qNXlKa0JEeUlTSnowSHBZRWg6VlVXWTd4Q0JPbDRSMC1KajFLQ2hKZw=="

# Global variables for lazy initialization
es_client = None
es_available = False

# Mock data for immediate functionality
MOCK_MOTIONS = [
    {
        "id": "motion_001",
        "name": "Walking Forward",
        "description": "Basic forward walking motion with natural arm swing",
        "motion_vector": [0.12, 0.34, 0.56, 0.78, 0.23, 0.45, 0.67, 0.89],
        "metadata": {
            "frames": 60,
            "joints": 24,
            "duration": 2.0,
            "format": "FBX",
            "category": "locomotion",
            "tags": ["walking", "forward", "basic"],
            "fps": 30.0,
            "file_size": 1024000
        },
        "created_at": "2025-10-09T12:00:00Z",
        "updated_at": "2025-10-09T12:00:00Z",
        "quality_score": 85.5,
        "motion_type": "locomotion",
        "complexity": 45.2,
        "blend_compatibility": "high",
        "file_path": "/motions/walking_forward.fbx",
        "checksum": "sha256_walking_hash"
    },
    {
        "id": "motion_002",
        "name": "Running Sprint",
        "description": "High-speed sprint running motion with dynamic arm movement",
        "motion_vector": [0.89, 0.67, 0.45, 0.23, 0.78, 0.56, 0.34, 0.12],
        "metadata": {
            "frames": 45,
            "joints": 24,
            "duration": 1.5,
            "format": "GLB",
            "category": "locomotion",
            "tags": ["running", "sprint", "fast"],
            "fps": 30.0,
            "file_size": 2048000
        },
        "created_at": "2025-10-09T12:01:00Z",
        "updated_at": "2025-10-09T12:01:00Z",
        "quality_score": 92.3,
        "motion_type": "locomotion",
        "complexity": 68.7,
        "blend_compatibility": "medium",
        "file_path": "/motions/running_sprint.glb",
        "checksum": "sha256_running_hash"
    },
    {
        "id": "motion_003",
        "name": "Dance Hip Hop",
        "description": "Energetic hip hop dance routine with rhythmic movements",
        "motion_vector": [0.55, 0.23, 0.78, 0.91, 0.34, 0.67, 0.12, 0.45],
        "metadata": {
            "frames": 180,
            "joints": 24,
            "duration": 6.0,
            "format": "TRC",
            "category": "dance",
            "tags": ["dance", "hip-hop", "rhythm"],
            "fps": 30.0,
            "file_size": 3072000
        },
        "created_at": "2025-10-09T12:02:00Z",
        "updated_at": "2025-10-09T12:02:00Z",
        "quality_score": 78.9,
        "motion_type": "dance",
        "complexity": 72.4,
        "blend_compatibility": "low",
        "file_path": "/motions/dance_hiphop.trc",
        "checksum": "sha256_dance_hash"
    }
]

def get_elasticsearch_client():
    """Lazy initialization of Elasticsearch client"""
    global es_client, es_available
    
    if es_client is not None:
        return es_client, es_available
    
    try:
        logger.info("Initializing Elasticsearch client...")
        from elasticsearch import Elasticsearch
        
        es_client = Elasticsearch(
            ES_CLOUD_URL,
            api_key=ES_API_KEY,
            request_timeout=10,  # Short timeout for quick response
            retry_on_timeout=False
        )
        
        # Quick connection test
        info = es_client.info()
        es_available = True
        logger.info(f"✅ Elasticsearch connected: {info['version']['number']}")
        
        return es_client, es_available
        
    except Exception as e:
        logger.warning(f"⚠️ Elasticsearch unavailable: {e}")
        es_available = False
        return None, es_available

def calculate_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    import math
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(a * a for a in vec2))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint - always responds quickly"""
    client, available = get_elasticsearch_client()
    
    return jsonify({
        "status": "healthy",
        "service": "MotionBlend Elasticsearch API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "elasticsearch": "available" if available else "mock_mode",
        "motions_loaded": len(MOCK_MOTIONS)
    })

@app.route('/search', methods=['POST'])
def search():
    """Vector similarity search"""
    try:
        data = request.get_json()
        query_vector = data.get('vector', [])
        size = data.get('size', 10)
        
        if len(query_vector) != 8:
            return jsonify({"error": "Vector must have 8 dimensions"}), 400
        
        client, available = get_elasticsearch_client()
        
        if available and client:
            try:
                # Elasticsearch k-NN search
                search_body = {
                    "size": size,
                    "query": {
                        "knn": {
                            "field": "motion_vector",
                            "query_vector": query_vector,
                            "k": size,
                            "num_candidates": size * 2
                        }
                    }
                }
                
                response = client.search(index=ES_INDEX_NAME, body=search_body, timeout='5s')
                
                results = []
                for hit in response['hits']['hits']:
                    motion = hit['_source']
                    motion['score'] = hit['_score']
                    results.append(motion)
                
                return jsonify({
                    "results": results,
                    "total": len(results),
                    "query_type": "elasticsearch_knn",
                    "elasticsearch": True
                })
                
            except Exception as e:
                logger.warning(f"Elasticsearch search failed: {e}")
                # Fall through to mock search
        
        # Mock vector search using cosine similarity
        scored_motions = []
        for motion in MOCK_MOTIONS:
            similarity = calculate_similarity(query_vector, motion['motion_vector'])
            motion_copy = motion.copy()
            motion_copy['score'] = similarity
            scored_motions.append(motion_copy)
        
        # Sort by similarity score (descending)
        scored_motions.sort(key=lambda x: x['score'], reverse=True)
        
        return jsonify({
            "results": scored_motions[:size],
            "total": len(scored_motions),
            "query_type": "mock_similarity",
            "elasticsearch": False
        })
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/search/text', methods=['POST'])
def search_text():
    """Text search"""
    try:
        data = request.get_json()
        query = data.get('query', '').lower()
        size = data.get('size', 10)
        
        if not query:
            return jsonify({"error": "Query text is required"}), 400
        
        client, available = get_elasticsearch_client()
        
        if available and client:
            try:
                # Elasticsearch text search
                search_body = {
                    "size": size,
                    "query": {
                        "multi_match": {
                            "query": query,
                            "fields": ["name^3", "description^2", "metadata.tags^2", "metadata.category"]
                        }
                    }
                }
                
                response = client.search(index=ES_INDEX_NAME, body=search_body, timeout='5s')
                
                results = []
                for hit in response['hits']['hits']:
                    motion = hit['_source']
                    motion['score'] = hit['_score']
                    results.append(motion)
                
                return jsonify({
                    "results": results,
                    "total": len(results),
                    "query": query,
                    "query_type": "elasticsearch_text",
                    "elasticsearch": True
                })
                
            except Exception as e:
                logger.warning(f"Elasticsearch text search failed: {e}")
                # Fall through to mock search
        
        # Mock text search
        results = []
        for motion in MOCK_MOTIONS:
            score = 0
            
            # Search in name (weight: 3)
            if query in motion['name'].lower():
                score += 3
            
            # Search in description (weight: 2)
            if query in motion['description'].lower():
                score += 2
            
            # Search in tags (weight: 2)
            for tag in motion['metadata']['tags']:
                if query in tag.lower():
                    score += 2
            
            # Search in category (weight: 1)
            if query in motion['metadata']['category'].lower():
                score += 1
            
            if score > 0:
                motion_copy = motion.copy()
                motion_copy['score'] = score
                results.append(motion_copy)
        
        # Sort by score (descending)
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return jsonify({
            "results": results[:size],
            "total": len(results),
            "query": query,
            "query_type": "mock_text",
            "elasticsearch": False
        })
        
    except Exception as e:
        logger.error(f"Text search error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/index/bulk', methods=['POST'])
def bulk_index():
    """Bulk index multiple motions"""
    try:
        data = request.get_json()
        documents = data.get('documents', [])
        
        if not documents:
            return jsonify({"error": "No documents provided"}), 400
        
        client, available = get_elasticsearch_client()
        
        if available and client:
            try:
                from elasticsearch import helpers
                
                # Prepare documents for bulk indexing
                bulk_docs = []
                for doc in documents:
                    bulk_docs.append({
                        "_index": ES_INDEX_NAME,
                        "_source": doc
                    })
                
                # Perform bulk indexing with short timeout
                bulk_response = helpers.bulk(
                    client.options(request_timeout=30),
                    bulk_docs,
                    index=ES_INDEX_NAME
                )
                
                return jsonify({
                    "success": True,
                    "indexed": bulk_response[0],
                    "errors": len(bulk_response[1]) if bulk_response[1] else 0,
                    "elasticsearch": True
                })
                
            except Exception as e:
                logger.warning(f"Elasticsearch bulk indexing failed: {e}")
                # Fall through to mock response
        
        # Mock response
        return jsonify({
            "success": True,
            "indexed": len(documents),
            "errors": 0,
            "elasticsearch": False,
            "note": "Documents processed in mock mode"
        })
        
    except Exception as e:
        logger.error(f"Bulk indexing error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/motions', methods=['GET'])
def get_motions():
    """Get all available motions (mock data)"""
    return jsonify(MOCK_MOTIONS)

@app.route('/status', methods=['GET'])
def status():
    """Detailed status endpoint"""
    client, available = get_elasticsearch_client()
    
    status_info = {
        "service": "MotionBlend AI API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "elasticsearch": {
            "available": available,
            "url": ES_CLOUD_URL if available else "not connected",
            "index": ES_INDEX_NAME
        },
        "mock_data": {
            "motions": len(MOCK_MOTIONS),
            "categories": list(set(m['metadata']['category'] for m in MOCK_MOTIONS))
        }
    }
    
    if available and client:
        try:
            # Get index stats
            stats = client.indices.stats(index=ES_INDEX_NAME)
            status_info["elasticsearch"]["document_count"] = stats['indices'][ES_INDEX_NAME]['total']['docs']['count']
        except:
            status_info["elasticsearch"]["document_count"] = "unknown"
    
    return jsonify(status_info)

if __name__ == '__main__':
    print("🚀 Starting MotionBlend AI Elasticsearch API")
    print("=" * 50)
    print(f"📊 Mock motions loaded: {len(MOCK_MOTIONS)}")
    print(f"🌐 Server starting on http://127.0.0.1:5002")
    print("⚡ Elasticsearch will be initialized on first request")
    print("✅ Ready for requests!")
    
    app.run(debug=True, host='127.0.0.1', port=5002)