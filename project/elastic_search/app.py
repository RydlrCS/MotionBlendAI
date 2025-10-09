#!/usr/bin/env python3
"""
MotionBlendAI Elasticsearch API Server
======================================

Comprehensive Flask application providing semantic search capabilities for motion capture data.
This server integrates with Elasticsearch Cloud to enable AI-powered motion discovery and blending.

Features:
---------
• Vector similarity search using dense vectors and k-NN
• Semantic text search with ELSER model integration
• Hybrid search combining vector and text similarity
• Bulk indexing with semantic field enhancement
• Comprehensive mock data for development and testing
• Production-ready error handling and fallback mechanisms
• CORS support for frontend integration
• Detailed health monitoring and status reporting

API Endpoints:
--------------
• GET  /health              - Health check with Elasticsearch status
• GET  /motions             - List all available mock motions
• POST /search              - Vector similarity search (k-NN)
• POST /search/text         - Text-based motion search
• POST /search/semantic     - Natural language semantic search
• POST /search/hybrid       - Combined vector + text search
• POST /index               - Index single motion document
• POST /index/bulk          - Bulk index multiple motions

Elasticsearch Integration:
-------------------------
The application connects to Elasticsearch Cloud using the latest cluster configuration:
• Cluster: my-elasticsearch-project-bb39cc.es.us-central1.gcp.elastic.cloud
• Index: motion-blend
• Features: Semantic text fields, vector search, bulk operations
• Timeout: 300s for semantic processing with ML models

Mock Data System:
----------------
Extensive mock motion database covering various categories:
• Athletic: Sports, fitness, explosive movements
• Dance: Hip-hop, contemporary, ballroom styles  
• Martial Arts: Karate, boxing, traditional forms
• Everyday: Walking, gestures, professional interactions
• Wellness: Yoga, meditation, therapeutic movements
• Performance: Theater, dramatic expressions
• Complex: Parkour, multi-skill sequences

Data Pipeline Integration:
-------------------------
Seamless integration with Fivetran connector for motion capture data ingestion:
• Automatic semantic field generation from motion characteristics
• Motion intensity analysis and intelligent categorization
• Dual indexing to BigQuery and Elasticsearch
• Support for BVH, FBX, TRC, and GLB motion formats

Development & Production:
------------------------
The application automatically detects Elasticsearch availability and falls back
to enhanced mock search when needed, ensuring consistent development experience.

Author: MotionBlendAI Team
Version: 2.0.0
License: MIT
Documentation: https://github.com/RydlrCS/MotionBlendAI
"""

from flask import Flask, request, jsonify
from typing import Dict, List, Any, Optional
import numpy as np
import logging

from ES_INDEX_NAME import ES_INDEX_NAME

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Elasticsearch configuration - Updated cluster with semantic text support
_DEFAULT_ES_API_KEY = "bHRLcXlaa0JSaHFSM2NuRk9tYVA6cDdxRWVUeGNScE9GWWRSNGo5VWlLZw=="  # New cluster API key
_DEFAULT_ES_CLOUD_URL = "https://my-elasticsearch-project-bb39cc.es.us-central1.gcp.elastic.cloud:443"  # New cluster URL

# Try to get from environment variables (fallback)
import os
ES_API_KEY = os.getenv('ES_API_KEY', _DEFAULT_ES_API_KEY)
ES_CLOUD_URL = os.getenv('ES_CLOUD_URL', _DEFAULT_ES_CLOUD_URL)

try:
    from flask_cors import CORS  # type: ignore
    flask_cors_available = True
except ImportError:
    flask_cors_available = False
    print("flask-cors not available")

try:
    from elasticsearch import Elasticsearch  # type: ignore
    elasticsearch_available = True
except ImportError:
    elasticsearch_available = False
    print("elasticsearch not available")
    Elasticsearch = None  # type: ignore

app = Flask(__name__)
if flask_cors_available:
    from flask_cors import CORS  # type: ignore
    CORS(app)  # Enable CORS for all routes

# Connect to Elasticsearch instance (cloud or local)
es: Optional[Any] = None
es_available: bool = False

def create_motion_mappings() -> Dict[str, Any]:
    """Define comprehensive field mappings for motion capture data."""
    return {
        "properties": {
            # Basic motion identification
            "id": {
                "type": "keyword"
            },
            "name": {
                "type": "text",
                "analyzer": "standard",
                "fields": {
                    "keyword": {
                        "type": "keyword",
                        "ignore_above": 256
                    },
                    "semantic": {
                        "type": "semantic_text"
                    }
                }
            },
            
            # Motion vector for similarity search
            "motion_vector": {
                "type": "dense_vector",
                "dims": 8,  # Adjust based on your vector dimensions
                "index": True,
                "similarity": "cosine"
            },
            
            # Metadata structure
            "metadata": {
                "properties": {
                    "frames": {
                        "type": "integer"
                    },
                    "joints": {
                        "type": "integer"
                    },
                    "duration": {
                        "type": "float"
                    },
                    "format": {
                        "type": "keyword"
                    },
                    "category": {
                        "type": "keyword",
                        "fields": {
                            "text": {
                                "type": "text",
                                "analyzer": "standard"
                            }
                        }
                    },
                    "tags": {
                        "type": "keyword",
                        "fields": {
                            "text": {
                                "type": "text",
                                "analyzer": "standard"
                            }
                        }
                    },
                    "fps": {
                        "type": "float"
                    },
                    "file_size": {
                        "type": "long"
                    }
                }
            },
            
            # Temporal data
            "created_at": {
                "type": "date"
            },
            "updated_at": {
                "type": "date"
            },
            
            # Quality metrics
            "quality_score": {
                "type": "float"
            },
            
            # Motion characteristics
            "motion_type": {
                "type": "keyword"
            },
            "complexity": {
                "type": "float"
            },
            
            # Semantic search fields
            "description": {
                "type": "text",
                "analyzer": "standard",
                "fields": {
                    "semantic": {
                        "type": "semantic_text"
                    }
                }
            },
            
            # Blend compatibility
            "blend_compatibility": {
                "type": "keyword"
            },
            
            # File information
            "file_path": {
                "type": "keyword",
                "index": False
            },
            "checksum": {
                "type": "keyword",
                "index": False
            }
        }
    }

def initialize_elasticsearch():
    """Initialize Elasticsearch connection and create index with mappings."""
    global es, es_available
    
    if not elasticsearch_available or Elasticsearch is None:
        print("Elasticsearch library not available")
        return
    
    try:
        # Connect to the new Elasticsearch cluster with semantic text support
        es = Elasticsearch(
            ES_CLOUD_URL,
            api_key=ES_API_KEY,
            verify_certs=True,
            request_timeout=300  # Extended timeout for semantic text operations
        )
        print(f"Attempting connection to Elasticsearch Cloud: {ES_CLOUD_URL}")
        
        # Test connection
        if es and es.ping():
            es_available = True
            cluster_info = es.info()
            print(f"✅ Connected to Elasticsearch {cluster_info['version']['number']}")
            
            # Create index if it doesn't exist
            if not es.indices.exists(index=ES_INDEX_NAME):
                mappings = {"mappings": create_motion_mappings()}
                es.indices.create(index=ES_INDEX_NAME, body=mappings)
                print(f"✅ Created index '{ES_INDEX_NAME}' with semantic text mappings")
            else:
                # Update existing index mappings
                try:
                    mappings = create_motion_mappings()
                    es.indices.put_mapping(index=ES_INDEX_NAME, body=mappings)
                    print(f"✅ Updated mappings for index '{ES_INDEX_NAME}'")
                except Exception as mapping_error:
                    print(f"⚠️ Mapping update: {mapping_error}")
        else:
            print("❌ Elasticsearch ping failed")
            es_available = False
            es = None
    
    except Exception as e:
        print(f"❌ Elasticsearch connection failed: {e}")
        es_available = False
        es = None


# Initialize Elasticsearch on startup
initialize_elasticsearch()

# Mock motion database for development - Comprehensive motion capture data
# This extensive database covers all major motion categories for testing and development
MOCK_MOTIONS: List[Dict[str, Any]] = [
    # === LOCOMOTION CATEGORY ===
    {
        "id": "motion_001",
        "name": "Walking Forward",
        "vector": [0.12, 0.34, 0.56, 0.78, 0.23, 0.45, 0.67, 0.89],
        "metadata": {
            "frames": 120,
            "joints": 25,
            "duration": 4.0,
            "format": "FBX",
            "category": "locomotion",
            "tags": ["walking", "forward", "basic", "everyday"]
        },
        "description": "Natural forward walking motion with steady pace and normal gait",
        "motion_type": "locomotion",
        "quality_score": 0.85,
        "complexity": 0.3,
        "created_at": "2025-10-09T10:00:00Z"
    },
    {
        "id": "motion_002", 
        "name": "Running Sprint",
        "vector": [0.23, 0.45, 0.12, 0.89, 0.34, 0.67, 0.56, 0.78],
        "metadata": {
            "frames": 90,
            "joints": 25,
            "duration": 3.0,
            "format": "GLB",
            "category": "locomotion",
            "tags": ["running", "sprint", "fast", "athletic"]
        },
        "description": "High-speed sprint running with explosive acceleration and dynamic movement",
        "motion_type": "athletic",
        "quality_score": 0.92,
        "complexity": 0.7,
        "created_at": "2025-10-09T10:05:00Z"
    },
    {
        "id": "motion_003",
        "name": "Casual Strolling",
        "vector": [0.15, 0.28, 0.42, 0.55, 0.31, 0.48, 0.62, 0.75],
        "metadata": {
            "frames": 150,
            "joints": 25,
            "duration": 5.0,
            "format": "BVH",
            "category": "locomotion",
            "tags": ["walking", "casual", "relaxed", "leisure"]
        },
        "description": "Relaxed casual walking with leisurely pace and natural arm swing",
        "motion_type": "locomotion",
        "quality_score": 0.78,
        "complexity": 0.25,
        "created_at": "2025-10-09T10:10:00Z"
    },
    
    # === DANCE CATEGORY ===
    {
        "id": "motion_004",
        "name": "Dance Hip Hop",
        "vector": [0.56, 0.12, 0.89, 0.23, 0.78, 0.34, 0.45, 0.67],
        "metadata": {
            "frames": 200,
            "joints": 30,
            "duration": 6.7,
            "format": "TRC",
            "category": "dance",
            "tags": ["dance", "hip-hop", "rhythm", "urban", "street"]
        },
        "description": "Dynamic hip-hop dance sequence with sharp movements and rhythmic beats",
        "motion_type": "dance",
        "quality_score": 0.88,
        "complexity": 0.75,
        "created_at": "2025-10-09T10:15:00Z"
    },
    {
        "id": "motion_005",
        "name": "Contemporary Dance Flow",
        "vector": [0.41, 0.63, 0.27, 0.85, 0.52, 0.39, 0.74, 0.68],
        "metadata": {
            "frames": 240,
            "joints": 32,
            "duration": 8.0,
            "format": "FBX",
            "category": "dance",
            "tags": ["contemporary", "flowing", "graceful", "artistic", "expressive"]
        },
        "description": "Flowing contemporary dance with graceful movements and artistic expression",
        "motion_type": "dance",
        "quality_score": 0.91,
        "complexity": 0.8,
        "created_at": "2025-10-09T10:20:00Z"
    },
    {
        "id": "motion_006",
        "name": "Ballroom Waltz",
        "vector": [0.33, 0.66, 0.44, 0.77, 0.29, 0.58, 0.71, 0.82],
        "metadata": {
            "frames": 180,
            "joints": 28,
            "duration": 6.0,
            "format": "GLB",
            "category": "dance",
            "tags": ["ballroom", "waltz", "elegant", "partner", "classical"]
        },
        "description": "Elegant ballroom waltz with refined posture and classical dance technique",
        "motion_type": "dance",
        "quality_score": 0.89,
        "complexity": 0.65,
        "created_at": "2025-10-09T10:25:00Z"
    },
    
    # === ATHLETIC CATEGORY ===
    {
        "id": "motion_007",
        "name": "Jumping High",
        "vector": [0.78, 0.23, 0.45, 0.12, 0.89, 0.56, 0.67, 0.34],
        "metadata": {
            "frames": 60,
            "joints": 25,
            "duration": 2.0,
            "format": "FBX",
            "category": "athletic",
            "tags": ["jumping", "vertical", "explosive", "athletic", "power"]
        },
        "description": "Explosive vertical jump with maximum height and athletic power",
        "motion_type": "athletic",
        "quality_score": 0.94,
        "complexity": 0.6,
        "created_at": "2025-10-09T10:30:00Z"
    },
    {
        "id": "motion_008",
        "name": "Basketball Layup",
        "vector": [0.87, 0.42, 0.65, 0.19, 0.93, 0.51, 0.76, 0.38],
        "metadata": {
            "frames": 75,
            "joints": 27,
            "duration": 2.5,
            "format": "GLB",
            "category": "athletic",
            "tags": ["basketball", "layup", "sports", "coordination", "skill"]
        },
        "description": "Professional basketball layup with precise ball handling and athletic coordination",
        "motion_type": "athletic",
        "quality_score": 0.96,
        "complexity": 0.85,
        "created_at": "2025-10-09T10:35:00Z"
    },
    {
        "id": "motion_009",
        "name": "Tennis Serve",
        "vector": [0.69, 0.31, 0.84, 0.47, 0.72, 0.26, 0.91, 0.53],
        "metadata": {
            "frames": 90,
            "joints": 26,
            "duration": 3.0,
            "format": "TRC",
            "category": "athletic",
            "tags": ["tennis", "serve", "precision", "technique", "professional"]
        },
        "description": "Professional tennis serve with perfect form and explosive power delivery",
        "motion_type": "athletic",
        "quality_score": 0.93,
        "complexity": 0.8,
        "created_at": "2025-10-09T10:40:00Z"
    },
    
    # === MARTIAL ARTS CATEGORY ===
    {
        "id": "motion_010",
        "name": "Boxing Jab",
        "vector": [0.34, 0.67, 0.23, 0.56, 0.12, 0.89, 0.78, 0.45],
        "metadata": {
            "frames": 45,
            "joints": 25,
            "duration": 1.5,
            "format": "NPY",
            "category": "combat",
            "tags": ["boxing", "punch", "martial-arts", "combat", "technique"]
        },
        "description": "Sharp boxing jab with precise form and controlled power delivery",
        "motion_type": "combat",
        "quality_score": 0.87,
        "complexity": 0.55,
        "created_at": "2025-10-09T10:45:00Z"
    },
    {
        "id": "motion_011",
        "name": "Karate Kata Form",
        "vector": [0.58, 0.73, 0.41, 0.86, 0.29, 0.64, 0.95, 0.37],
        "metadata": {
            "frames": 300,
            "joints": 30,
            "duration": 10.0,
            "format": "FBX",
            "category": "martial-arts",
            "tags": ["karate", "kata", "traditional", "discipline", "form"]
        },
        "description": "Traditional karate kata with precise movements and martial discipline",
        "motion_type": "combat",
        "quality_score": 0.95,
        "complexity": 0.9,
        "created_at": "2025-10-09T10:50:00Z"
    },
    {
        "id": "motion_012",
        "name": "Tai Chi Flow",
        "vector": [0.22, 0.55, 0.38, 0.71, 0.46, 0.83, 0.17, 0.94],
        "metadata": {
            "frames": 420,
            "joints": 28,
            "duration": 14.0,
            "format": "BVH",
            "category": "martial-arts",
            "tags": ["tai-chi", "meditative", "flowing", "balance", "wellness"]
        },
        "description": "Meditative tai chi sequence with flowing movements and internal focus",
        "motion_type": "wellness",
        "quality_score": 0.91,
        "complexity": 0.7,
        "created_at": "2025-10-09T10:55:00Z"
    },
    
    # === WELLNESS CATEGORY ===
    {
        "id": "motion_013",
        "name": "Yoga Pose Flow",
        "vector": [0.45, 0.78, 0.34, 0.67, 0.56, 0.12, 0.23, 0.89],
        "metadata": {
            "frames": 180,
            "joints": 30,
            "duration": 6.0,
            "format": "GLB",
            "category": "wellness",
            "tags": ["yoga", "flexibility", "meditation", "mindful", "peaceful"]
        },
        "description": "Peaceful yoga flow sequence with mindful breathing and flexibility training",
        "motion_type": "wellness",
        "quality_score": 0.89,
        "complexity": 0.5,
        "created_at": "2025-10-09T11:00:00Z"
    },
    {
        "id": "motion_014",
        "name": "Stretching Routine",
        "vector": [0.31, 0.64, 0.48, 0.75, 0.39, 0.82, 0.26, 0.97],
        "metadata": {
            "frames": 240,
            "joints": 26,
            "duration": 8.0,
            "format": "TRC",
            "category": "wellness",
            "tags": ["stretching", "flexibility", "recovery", "therapeutic", "gentle"]
        },
        "description": "Comprehensive stretching routine for flexibility and muscle recovery",
        "motion_type": "wellness",
        "quality_score": 0.84,
        "complexity": 0.4,
        "created_at": "2025-10-09T11:05:00Z"
    },
    
    # === GESTURE CATEGORY ===
    {
        "id": "motion_015",
        "name": "Professional Handshake",
        "vector": [0.42, 0.68, 0.35, 0.79, 0.51, 0.24, 0.86, 0.63],
        "metadata": {
            "frames": 60,
            "joints": 22,
            "duration": 2.0,
            "format": "FBX",
            "category": "gesture",
            "tags": ["handshake", "professional", "business", "greeting", "confident"]
        },
        "description": "Confident professional handshake with proper business etiquette",
        "motion_type": "gesture",
        "quality_score": 0.82,
        "complexity": 0.3,
        "created_at": "2025-10-09T11:10:00Z"
    },
    {
        "id": "motion_016",
        "name": "Presenting Gesture",
        "vector": [0.59, 0.36, 0.81, 0.47, 0.73, 0.28, 0.92, 0.54],
        "metadata": {
            "frames": 90,
            "joints": 24,
            "duration": 3.0,
            "format": "GLB",
            "category": "gesture",
            "tags": ["presenting", "demonstration", "professional", "communication", "expressive"]
        },
        "description": "Professional presentation gesture with clear communication and confident posture",
        "motion_type": "gesture",
        "quality_score": 0.86,
        "complexity": 0.45,
        "created_at": "2025-10-09T11:15:00Z"
    },
    
    # === PERFORMANCE CATEGORY ===
    {
        "id": "motion_017",
        "name": "Theater Dramatic Pose",
        "vector": [0.76, 0.43, 0.89, 0.32, 0.65, 0.57, 0.94, 0.41],
        "metadata": {
            "frames": 120,
            "joints": 28,
            "duration": 4.0,
            "format": "BVH",
            "category": "performance",
            "tags": ["theater", "dramatic", "stage", "acting", "expressive"]
        },
        "description": "Dramatic theatrical pose with exaggerated expression for stage performance",
        "motion_type": "gesture",
        "quality_score": 0.88,
        "complexity": 0.7,
        "created_at": "2025-10-09T11:20:00Z"
    },
    {
        "id": "motion_018",
        "name": "Musical Conducting",
        "vector": [0.38, 0.91, 0.57, 0.74, 0.25, 0.83, 0.46, 0.69],
        "metadata": {
            "frames": 200,
            "joints": 26,
            "duration": 6.7,
            "format": "TRC",
            "category": "performance",
            "tags": ["conducting", "musical", "rhythm", "orchestral", "precise"]
        },
        "description": "Musical conducting with precise rhythm and expressive arm movements",
        "motion_type": "gesture",
        "quality_score": 0.93,
        "complexity": 0.8,
        "created_at": "2025-10-09T11:25:00Z"
    },
    
    # === COMPLEX MULTI-SKILL CATEGORY ===
    {
        "id": "motion_019",
        "name": "Parkour Sequence",
        "vector": [0.94, 0.67, 0.85, 0.52, 0.91, 0.38, 0.76, 0.83],
        "metadata": {
            "frames": 360,
            "joints": 32,
            "duration": 12.0,
            "format": "FBX",
            "category": "athletic",
            "tags": ["parkour", "obstacles", "vaulting", "climbing", "complex", "fluid"]
        },
        "description": "Complex parkour sequence with fluid transitions between multiple obstacles",
        "motion_type": "athletic",
        "quality_score": 0.97,
        "complexity": 0.95,
        "created_at": "2025-10-09T11:30:00Z"
    },
    {
        "id": "motion_020",
        "name": "Acrobatic Routine",
        "vector": [0.88, 0.54, 0.92, 0.37, 0.79, 0.65, 0.96, 0.41],
        "metadata": {
            "frames": 300,
            "joints": 30,
            "duration": 10.0,
            "format": "GLB",
            "category": "athletic",
            "tags": ["acrobatic", "gymnastics", "flips", "coordination", "spectacular"]
        },
        "description": "Spectacular acrobatic routine with flips, spins, and gymnastic coordination",
        "motion_type": "athletic",
        "quality_score": 0.95,
        "complexity": 0.92,
        "created_at": "2025-10-09T11:35:00Z"
    }
]

def calculate_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    try:
        v1 = np.array(vec1, dtype=np.float64)
        v2 = np.array(vec2, dtype=np.float64)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    except Exception:
        return 0.0

@app.route('/search', methods=['POST'])
def semantic_search():
    """
    Perform a k-NN vector search on the 'motion-blend' index using the provided vector.
    Expects JSON: {"vector": [...], "k": 10}
    Returns: List of matching documents with similarity scores.
    """
    try:
        req = request.get_json()
        if not req:
            return jsonify({"error": "Invalid JSON"}), 400
            
        query_vector = req.get("vector")
        k = req.get("k", 10)
        
        if not query_vector or not isinstance(query_vector, list):
            return jsonify({"error": "Vector field is required and must be a list"}), 400
        
        # Validate vector contains numbers
        try:
            vector_floats = [float(x) for x in query_vector]  # type: ignore
        except (ValueError, TypeError):
            return jsonify({"error": "Vector must contain only numbers"}), 400
        
        hits: List[Dict[str, Any]] = []
        
        if es_available and es:
            # Use real Elasticsearch with k-NN search
            try:
                response = es.search(
                    index=ES_INDEX_NAME,
                    body={
                        "size": k,
                        "query": {
                            "knn": {
                                "field": "motion_vector",
                                "query_vector": vector_floats,
                                "k": k,
                                "num_candidates": k * 2
                            }
                        },
                        "_source": {
                            "excludes": ["motion_vector"]  # Exclude large vector from response
                        }
                    }
                )
                
                hits = []
                for hit in response["hits"]["hits"]:
                    motion_data = hit["_source"]
                    motion_data["similarity_score"] = hit["_score"]
                    motion_data["id"] = hit["_id"]
                    hits.append(motion_data)
                    
            except Exception as e:
                print(f"Elasticsearch search error: {e}")
                # Fallback to mock data
                hits = _mock_vector_search(vector_floats, k)
        else:
            # Use mock data with similarity calculation
            hits = _mock_vector_search(vector_floats, k)
        
        return jsonify(hits)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def _mock_vector_search(vector_floats: List[float], k: int) -> List[Dict[str, Any]]:
    """Mock vector search using MOCK_MOTIONS data."""
    results: List[Dict[str, Any]] = []
    for motion in MOCK_MOTIONS:
        similarity = calculate_similarity(vector_floats, motion["vector"])  # type: ignore
        motion_copy = motion.copy()
        motion_copy["similarity_score"] = similarity
        results.append(motion_copy)
    
    # Sort by similarity and take top k
    results.sort(key=lambda x: x["similarity_score"], reverse=True)  # type: ignore
    return results[:k]

@app.route('/search/text', methods=['POST'])
def text_search():
    """
    Perform text-based search on motion names, tags, and semantic content.
    Expects JSON: {"query": "search text", "k": 10}
    Returns: List of matching documents.
    """
    try:
        req = request.get_json()
        if not req:
            return jsonify({"error": "Invalid JSON"}), 400
            
        query_text = str(req.get("query", "")).lower()
        k = int(req.get("k", 10))
        
        if not query_text:
            return jsonify([])
        
        hits: List[Dict[str, Any]] = []
        
        if es_available and es:
            # Use Elasticsearch with semantic search
            try:
                response = es.search(
                    index=ES_INDEX_NAME,
                    body={
                        "size": k,
                        "query": {
                            "bool": {
                                "should": [
                                    # Semantic text search
                                    {
                                        "semantic": {
                                            "field": "name.semantic",
                                            "query": query_text
                                        }
                                    },
                                    {
                                        "semantic": {
                                            "field": "description.semantic",
                                            "query": query_text
                                        }
                                    },
                                    # Traditional text search
                                    {
                                        "multi_match": {
                                            "query": query_text,
                                            "fields": [
                                                "name^3",
                                                "metadata.tags^2",
                                                "metadata.category^2",
                                                "description"
                                            ],
                                            "type": "best_fields",
                                            "fuzziness": "AUTO"
                                        }
                                    },
                                    # Exact keyword matches
                                    {
                                        "terms": {
                                            "metadata.tags": [query_text]
                                        }
                                    },
                                    {
                                        "term": {
                                            "metadata.category": query_text
                                        }
                                    }
                                ],
                                "minimum_should_match": 1
                            }
                        },
                        "highlight": {
                            "fields": {
                                "name": {},
                                "description": {},
                                "metadata.tags": {}
                            }
                        }
                    }
                )
                
                hits = []
                for hit in response["hits"]["hits"]:
                    motion_data = hit["_source"]
                    motion_data["similarity_score"] = hit["_score"] / 10.0  # Normalize score
                    motion_data["id"] = hit["_id"]
                    if "highlight" in hit:
                        motion_data["highlight"] = hit["highlight"]
                    hits.append(motion_data)
                    
            except Exception as e:
                print(f"Elasticsearch text search error: {e}")
                # Fallback to mock data
                hits = _mock_text_search(query_text, k)
        else:
            # Use mock data
            hits = _mock_text_search(query_text, k)
        
        return jsonify(hits)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def _mock_text_search(query_text: str, k: int) -> List[Dict[str, Any]]:
    """Mock text search using MOCK_MOTIONS data."""
    matches: List[Dict[str, Any]] = []
    for motion in MOCK_MOTIONS:
        score = 0.0
        
        # Check name match
        motion_name = str(motion["name"]).lower()
        if query_text in motion_name:
            score += 1.0
        
        # Check tag matches
        motion_tags = motion["metadata"]["tags"]
        if isinstance(motion_tags, list):
            for tag in motion_tags:  # type: ignore
                tag_str = str(tag).lower()  # type: ignore
                if query_text in tag_str:
                    score += 0.5
        
        # Check category match
        motion_category = str(motion["metadata"]["category"]).lower()
        if query_text in motion_category:
            score += 0.7
        
        if score > 0:
            motion_copy = motion.copy()
            motion_copy["similarity_score"] = min(score, 1.0)  # Cap at 1.0
            matches.append(motion_copy)
    
    # Sort by score and return top k
    matches.sort(key=lambda x: float(x["similarity_score"]), reverse=True)  # type: ignore
    return matches[:k]

@app.route('/index', methods=['POST'])
def index_motion():
    """
    Index a motion document to Elasticsearch.
    Expects JSON with motion data including vector and metadata.
    """
    try:
        req = request.get_json()
        if not req:
            return jsonify({"error": "Invalid JSON"}), 400
        
        if es_available and es:
            # Index to Elasticsearch
            doc_id = req.get("id")
            response = es.index(
                index=ES_INDEX_NAME,
                id=doc_id,
                body=req
            )
            return jsonify({
                "success": True,
                "id": response["_id"],
                "result": response["result"]
            })
        else:
            return jsonify({"error": "Elasticsearch not available"}), 503
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with Elasticsearch status."""
    health_data: Dict[str, Any] = {
        "status": "healthy",
        "elasticsearch_available": es_available,
        "mock_motions_count": len(MOCK_MOTIONS),
        "index_name": ES_INDEX_NAME
    }
    
    if es_available and es:
        try:
            # Get Elasticsearch cluster health
            cluster_health = es.cluster.health()
            index_stats = es.indices.stats(index=ES_INDEX_NAME)
            
            health_data["elasticsearch"] = {
                "cluster_status": cluster_health["status"],
                "number_of_nodes": cluster_health["number_of_nodes"],
                "index_docs_count": index_stats["indices"][ES_INDEX_NAME]["total"]["docs"]["count"],
                "index_size": index_stats["indices"][ES_INDEX_NAME]["total"]["store"]["size_in_bytes"]
            }
        except Exception as e:
            health_data["elasticsearch_error"] = str(e)
    
    return jsonify(health_data)

@app.route('/search/semantic', methods=['POST'])
def semantic_text_search():
    """
    Advanced semantic search using natural language queries.
    Expects JSON: {"query": "natural language query", "k": 10}
    Returns: List of semantically matching documents.
    """
    try:
        req = request.get_json()
        if not req:
            return jsonify({"error": "Invalid JSON"}), 400
            
        query_text = str(req.get("query", "")).strip()
        k = int(req.get("k", 10))
        
        if not query_text:
            return jsonify({"error": "Query text is required"}), 400
        
        hits: List[Dict[str, Any]] = []
        
        if es_available and es:
            try:
                # Use semantic text search with ELSER model
                response = es.search(
                    index=ES_INDEX_NAME,
                    body={
                        "size": k,
                        "query": {
                            "bool": {
                                "should": [
                                    # Primary semantic search
                                    {
                                        "semantic": {
                                            "field": "name.semantic",
                                            "query": query_text
                                        }
                                    },
                                    {
                                        "semantic": {
                                            "field": "description.semantic", 
                                            "query": query_text
                                        }
                                    },
                                    # Boosted exact matches
                                    {
                                        "match_phrase": {
                                            "name": {
                                                "query": query_text,
                                                "boost": 3.0
                                            }
                                        }
                                    },
                                    # Category and tag matches
                                    {
                                        "multi_match": {
                                            "query": query_text,
                                            "fields": [
                                                "metadata.category^2",
                                                "metadata.tags^1.5",
                                                "motion_type^1.5"
                                            ],
                                            "type": "best_fields",
                                            "fuzziness": "AUTO"
                                        }
                                    }
                                ],
                                "minimum_should_match": 1
                            }
                        },
                        "_source": {
                            "excludes": ["motion_vector"]  # Exclude large vectors
                        },
                        "highlight": {
                            "fields": {
                                "name": {},
                                "description": {},
                                "metadata.tags": {},
                                "metadata.category": {}
                            }
                        }
                    }
                )
                
                hits = []
                for hit in response["hits"]["hits"]:
                    motion_data = hit["_source"]
                    motion_data["semantic_score"] = hit["_score"]
                    motion_data["id"] = hit["_id"]
                    if "highlight" in hit:
                        motion_data["highlight"] = hit["highlight"]
                    hits.append(motion_data)
                    
            except Exception as e:
                print(f"Semantic search error: {e}")
                # Fallback to enhanced mock search
                hits = _enhanced_mock_search(query_text, k)
        else:
            hits = _enhanced_mock_search(query_text, k)
        
        return jsonify({
            "query": query_text,
            "results": hits,
            "total": len(hits),
            "semantic_search": True
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def _enhanced_mock_search(query_text: str, k: int) -> List[Dict[str, Any]]:
    """Enhanced mock search with semantic-like scoring."""
    import re
    
    matches: List[Dict[str, Any]] = []
    query_words = re.findall(r'\\w+', query_text.lower())
    
    for motion in MOCK_MOTIONS:
        score = 0.0
        
        # Name matching with word overlap
        motion_name = str(motion["name"]).lower()
        name_words = re.findall(r'\\w+', motion_name)
        name_overlap = len(set(query_words) & set(name_words))
        if name_overlap > 0:
            score += (name_overlap / len(query_words)) * 2.0
        
        # Tag matching
        motion_tags = motion["metadata"]["tags"]
        if isinstance(motion_tags, list):
            for tag in motion_tags:  # type: ignore
                tag_words = re.findall(r'\\w+', str(tag).lower())  # type: ignore
                tag_overlap = len(set(query_words) & set(tag_words))
                if tag_overlap > 0:
                    score += (tag_overlap / len(query_words)) * 1.5
        
        # Category matching
        motion_category = str(motion["metadata"]["category"]).lower()
        category_words = re.findall(r'\\w+', motion_category)
        category_overlap = len(set(query_words) & set(category_words))
        if category_overlap > 0:
            score += (category_overlap / len(query_words)) * 1.0
        
        # Semantic-like scoring for related concepts
        semantic_map = {
            "walk": ["locomotion", "movement", "step"],
            "run": ["sprint", "fast", "athletic"],
            "dance": ["rhythm", "music", "performance"],
            "jump": ["leap", "athletic", "explosive"],
            "fight": ["combat", "martial", "boxing"],
            "yoga": ["wellness", "flexibility", "meditation"]
        }
        
        for query_word in query_words:
            if query_word in semantic_map:
                related_words = semantic_map[query_word]
                all_motion_text = f"{motion_name} {' '.join(motion_tags)} {motion_category}".lower()  # type: ignore
                for related_word in related_words:
                    if related_word in all_motion_text:
                        score += 0.5
        
        if score > 0:
            motion_copy = motion.copy()
            motion_copy["semantic_score"] = min(score, 3.0)  # Cap at 3.0
            matches.append(motion_copy)
    
    # Sort by semantic score and return top k
    matches.sort(key=lambda x: float(x["semantic_score"]), reverse=True)  # type: ignore
    return matches[:k]

@app.route('/search/hybrid', methods=['POST'])
def hybrid_search():
    """
    Hybrid search combining vector similarity and semantic text search.
    Expects JSON: {"vector": [...], "query": "text", "k": 10, "vector_weight": 0.5}
    Returns: Combined results with weighted scoring.
    """
    try:
        req = request.get_json()
        if not req:
            return jsonify({"error": "Invalid JSON"}), 400
            
        query_vector = req.get("vector")
        query_text = str(req.get("query", "")).strip()
        k = int(req.get("k", 10))
        vector_weight = float(req.get("vector_weight", 0.5))  # 0.5 = equal weight
        text_weight = 1.0 - vector_weight
        
        if not query_vector and not query_text:
            return jsonify({"error": "Either vector or query text is required"}), 400
        
        hits: List[Dict[str, Any]] = []
        
        if es_available and es:
            try:
                # Build hybrid query
                should_queries = []
                
                if query_vector:
                    try:
                        vector_floats = [float(x) for x in query_vector]  # type: ignore
                        should_queries.append({
                            "knn": {
                                "field": "motion_vector",
                                "query_vector": vector_floats,
                                "k": k * 2,
                                "num_candidates": k * 4,
                                "boost": vector_weight
                            }
                        })
                    except (ValueError, TypeError):
                        return jsonify({"error": "Invalid vector format"}), 400
                
                if query_text:
                    should_queries.extend([
                        {
                            "semantic": {
                                "field": "name.semantic",
                                "query": query_text,
                                "boost": text_weight * 2.0
                            }
                        },
                        {
                            "semantic": {
                                "field": "description.semantic",
                                "query": query_text,
                                "boost": text_weight * 1.5
                            }
                        },
                        {
                            "multi_match": {
                                "query": query_text,
                                "fields": [
                                    "name^3",
                                    "metadata.tags^2", 
                                    "metadata.category^2"
                                ],
                                "type": "best_fields",
                                "fuzziness": "AUTO",
                                "boost": text_weight
                            }
                        }
                    ])
                
                query_body = {
                    "size": k,
                    "query": {
                        "bool": {
                            "should": should_queries,
                            "minimum_should_match": 1
                        }
                    },
                    "_source": {
                        "excludes": ["motion_vector"]
                    }
                }
                
                if query_text:
                    query_body["highlight"] = {
                        "fields": {
                            "name": {},
                            "description": {},
                            "metadata.tags": {}
                        }
                    }
                
                response = es.search(index=ES_INDEX_NAME, body=query_body)
                
                hits = []
                for hit in response["hits"]["hits"]:
                    motion_data = hit["_source"]
                    motion_data["hybrid_score"] = hit["_score"]
                    motion_data["id"] = hit["_id"]
                    if "highlight" in hit:
                        motion_data["highlight"] = hit["highlight"]
                    hits.append(motion_data)
                    
            except Exception as e:
                print(f"Hybrid search error: {e}")
                # Fallback to mock hybrid search
                hits = _mock_hybrid_search(query_vector, query_text, k, vector_weight)
        else:
            hits = _mock_hybrid_search(query_vector, query_text, k, vector_weight)
        
        return jsonify({
            "query_vector": bool(query_vector),
            "query_text": query_text,
            "results": hits,
            "total": len(hits),
            "weights": {
                "vector": vector_weight,
                "text": text_weight
            },
            "hybrid_search": True
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def _mock_hybrid_search(query_vector: Optional[List[float]], query_text: str, k: int, vector_weight: float) -> List[Dict[str, Any]]:
    """Mock hybrid search combining vector and text similarity."""
    text_weight = 1.0 - vector_weight
    scored_motions: List[Dict[str, Any]] = []
    
    for motion in MOCK_MOTIONS:
        total_score = 0.0
        
        # Vector similarity component
        if query_vector:
            try:
                vector_floats = [float(x) for x in query_vector]  # type: ignore
                vector_sim = calculate_similarity(vector_floats, motion["vector"])  # type: ignore
                total_score += vector_sim * vector_weight
            except (ValueError, TypeError):
                pass
        
        # Text similarity component
        if query_text:
            text_results = _enhanced_mock_search(query_text, len(MOCK_MOTIONS))
            text_sim = 0.0
            for result in text_results:
                if result["id"] == motion["id"]:
                    text_sim = result.get("semantic_score", 0.0) / 3.0  # Normalize to 0-1
                    break
            total_score += text_sim * text_weight
        
        if total_score > 0:
            motion_copy = motion.copy()
            motion_copy["hybrid_score"] = total_score
            scored_motions.append(motion_copy)
    
    # Sort by hybrid score and return top k
    scored_motions.sort(key=lambda x: float(x["hybrid_score"]), reverse=True)  # type: ignore
    return scored_motions[:k]

@app.route('/index/bulk', methods=['POST'])
def bulk_index():
    """Bulk index multiple motions with semantic text support"""
    try:
        data = request.get_json()
        
        # Handle both array format and object with documents key
        if isinstance(data, list):
            documents = data
        elif isinstance(data, dict) and 'documents' in data:
            documents = data.get('documents', [])
        else:
            return jsonify({"error": "Expected JSON array or object with 'documents' key"}), 400
        
        if not documents:
            return jsonify({"error": "No documents provided"}), 400
        
        if es_available and es:
            # Prepare documents for bulk indexing with semantic fields
            bulk_docs = []
            for doc in documents:
                # Ensure semantic text fields are present
                if "name" in doc and "semantic" not in str(doc.get("name", {})):
                    if isinstance(doc["name"], str):
                        doc["name"] = {
                            "text": doc["name"],
                            "semantic": doc["name"]
                        }
                
                if "description" in doc and isinstance(doc["description"], str):
                    doc["description"] = {
                        "text": doc["description"],
                        "semantic": doc["description"]
                    }
                
                bulk_docs.append({
                    "_index": ES_INDEX_NAME,
                    "_source": doc
                })
            
            # Perform bulk indexing with extended timeout for semantic processing
            try:
                from elasticsearch import helpers  # type: ignore
                bulk_response = helpers.bulk(
                    es,  # type: ignore
                    bulk_docs,
                    index=ES_INDEX_NAME,
                    request_timeout=300  # 5 minutes for semantic text processing
                )
                
                return jsonify({
                    "success": True,
                    "indexed": bulk_response[0] if bulk_response else len(documents),
                    "errors": len(bulk_response[1]) if bulk_response and len(bulk_response) > 1 else 0,
                    "details": bulk_response[1][:5] if bulk_response and len(bulk_response) > 1 and bulk_response[1] else [],
                    "semantic_processing": True
                })
            except Exception as bulk_error:
                return jsonify({"error": f"Bulk indexing failed: {bulk_error}"}), 500
        else:
            # Mock response - simulate successful indexing
            return jsonify({
                "success": True,
                "indexed": len(documents),
                "errors": 0,
                "mode": "mock",
                "semantic_processing": False
            })
            
    except Exception as e:
        logger.error(f"Bulk indexing error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/motions', methods=['GET'])
def get_motions():
    """Get all available motions (mock data)"""
    return jsonify(MOCK_MOTIONS)

if __name__ == '__main__':
    print(f"Starting Elasticsearch API server...")
    print(f"Elasticsearch available: {es_available}")
    print(f"Mock motions loaded: {len(MOCK_MOTIONS)}")
    print(f"Index name: {ES_INDEX_NAME}")
    if es_available:
        print("✅ Connected to Elasticsearch - using real search")
    else:
        print("⚠️  Using mock data for development")
    app.run(debug=True, host='127.0.0.1', port=5002)
