from flask import Flask, request, jsonify
from typing import Dict, List, Any, Optional
import numpy as np
import os

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

app = Flask(__name__)
if flask_cors_available:
    from flask_cors import CORS  # type: ignore
    CORS(app)  # Enable CORS for all routes

# Elasticsearch configuration
ES_CLOUD_URL = os.getenv('ES_CLOUD_URL', 'https://my-elasticsearch-project-ba986d.es.us-central1.gcp.elastic.cloud:443')
ES_API_KEY = os.getenv('ES_API_KEY', 'S21qNXlKa0JEeUlTSnowSHBZRWg6VlVXWTd4Q0JPbDRSMC1KajFLQ2hKZw==')
ES_INDEX_NAME = "motion-blend"

# Connect to Elasticsearch instance (cloud or local)
es: Optional[Any] = None
es_available = False

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
    
    if not elasticsearch_available:
        print("Elasticsearch library not available")
        return
    
    try:
        # Try cloud connection first
        if ES_API_KEY and ES_CLOUD_URL:
            es = Elasticsearch(
                ES_CLOUD_URL,
                api_key=ES_API_KEY,
                verify_certs=True
            )
            print(f"Attempting connection to Elasticsearch Cloud: {ES_CLOUD_URL}")
        else:
            # Fallback to local connection
            es = Elasticsearch([{"host": "localhost", "port": 9200}])
            print("Attempting connection to local Elasticsearch")
        
        # Test connection
        es_available = es.ping()
        
        if es_available:
            print("✅ Elasticsearch connection successful")
            
            # Create index with mappings if it doesn't exist
            if not es.indices.exists(index=ES_INDEX_NAME):
                mappings = create_motion_mappings()
                
                index_config = {
                    "mappings": mappings,
                    "settings": {
                        "number_of_shards": 1,
                        "number_of_replicas": 0,
                        "analysis": {
                            "analyzer": {
                                "motion_analyzer": {
                                    "type": "standard",
                                    "stopwords": "_english_"
                                }
                            }
                        }
                    }
                }
                
                response = es.indices.create(index=ES_INDEX_NAME, body=index_config)
                print(f"✅ Created index '{ES_INDEX_NAME}' with mappings")
                print(f"Response: {response}")
            else:
                # Update mappings if index exists
                mappings = create_motion_mappings()
                response = es.indices.put_mapping(index=ES_INDEX_NAME, body=mappings)
                print(f"✅ Updated mappings for index '{ES_INDEX_NAME}'")
                print(f"Mapping response: {response}")
                
        else:
            print("❌ Elasticsearch ping failed")
            
    except Exception as e:
        print(f"❌ Elasticsearch connection failed: {e}")
        es_available = False
        es = None

# Initialize Elasticsearch on startup
initialize_elasticsearch()

# Mock motion database for development
MOCK_MOTIONS: List[Dict[str, Any]] = [
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
            "tags": ["walking", "forward", "basic"]
        }
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
            "tags": ["running", "sprint", "fast"]
        }
    },
    {
        "id": "motion_003",
        "name": "Dance Hip Hop",
        "vector": [0.56, 0.12, 0.89, 0.23, 0.78, 0.34, 0.45, 0.67],
        "metadata": {
            "frames": 200,
            "joints": 30,
            "duration": 6.7,
            "format": "TRC",
            "category": "dance",
            "tags": ["dance", "hip-hop", "rhythm"]
        }
    },
    {
        "id": "motion_004",
        "name": "Jumping High",
        "vector": [0.78, 0.23, 0.45, 0.12, 0.89, 0.56, 0.67, 0.34],
        "metadata": {
            "frames": 60,
            "joints": 25,
            "duration": 2.0,
            "format": "FBX",
            "category": "athletic",
            "tags": ["jumping", "vertical", "explosive"]
        }
    },
    {
        "id": "motion_005",
        "name": "Boxing Jab",
        "vector": [0.34, 0.67, 0.23, 0.56, 0.12, 0.89, 0.78, 0.45],
        "metadata": {
            "frames": 45,
            "joints": 25,
            "duration": 1.5,
            "format": "NPY",
            "category": "combat",
            "tags": ["boxing", "punch", "martial-arts"]
        }
    },
    {
        "id": "motion_006",
        "name": "Yoga Pose Flow",
        "vector": [0.45, 0.78, 0.34, 0.67, 0.56, 0.12, 0.23, 0.89],
        "metadata": {
            "frames": 180,
            "joints": 30,
            "duration": 6.0,
            "format": "GLB",
            "category": "wellness",
            "tags": ["yoga", "flexibility", "meditation"]
        }
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

@app.route('/index/bulk', methods=['POST'])
def bulk_index_motions():
    """
    Bulk index multiple motion documents.
    Expects JSON array of motion documents.
    """
    try:
        req = request.get_json()
        if not req or not isinstance(req, list):
            return jsonify({"error": "Invalid JSON - expected array"}), 400
        
        if es_available and es:
            from elasticsearch.helpers import bulk  # type: ignore
            
            # Prepare documents for bulk indexing
            docs = []
            for motion in req:
                docs.append({
                    "_index": ES_INDEX_NAME,
                    "_id": motion.get("id"),
                    "_source": motion
                })
            
            # Perform bulk indexing
            success_count, failed_items = bulk(es, docs)
            
            return jsonify({
                "success": True,
                "indexed": success_count,
                "failed": len(failed_items) if failed_items else 0,
                "errors": failed_items if failed_items else []
            })
        else:
            return jsonify({"error": "Elasticsearch not available"}), 503
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with Elasticsearch status."""
    health_data = {
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

@app.route('/index/bulk', methods=['POST'])
def bulk_index():
    """Bulk index multiple motions"""
    try:
        data = request.get_json()
        documents = data.get('documents', [])
        
        if not documents:
            return jsonify({"error": "No documents provided"}), 400
        
        if es_available:
            # Prepare documents for bulk indexing
            bulk_docs = []
            for doc in documents:
                bulk_docs.append({
                    "_index": ES_INDEX_NAME,
                    "_source": doc
                })
            
            # Perform bulk indexing with timeout
            from elasticsearch import helpers
            bulk_response = helpers.bulk(
                es.options(request_timeout=300),
                bulk_docs,
                index=ES_INDEX_NAME
            )
            
            return jsonify({
                "success": True,
                "indexed": bulk_response[0],
                "errors": len(bulk_response[1]) if bulk_response[1] else 0,
                "details": bulk_response[1][:5] if bulk_response[1] else []
            })
        else:
            # Mock response
            return jsonify({
                "success": True,
                "indexed": len(documents),
                "errors": 0,
                "mode": "mock"
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
