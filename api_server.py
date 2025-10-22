#!/usr/bin/env python3
"""
MotionBlend AI Production API Server
Implements all endpoints required by the React UI
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import logging
import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Enable CORS for all origins (restrict in production)
CORS(app, resources={
    r"/*": {
        "origins": ["*"],  # Change to specific domains in production
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Configuration
GCS_BUCKET = os.environ.get('GCS_BUCKET', 'motionblend-mocap')
BQ_PROJECT = os.environ.get('BQ_PROJECT', 'motionblend-ai')
BQ_DATASET = os.environ.get('BQ_DATASET', 'RAW_DEV')
ES_URL = os.environ.get('ELASTICSEARCH_URL', 'https://elasticsearch-motionblend-ba986d.es.us-central1.gcp.elastic.cloud')
ES_API_KEY = os.environ.get('ES_API_KEY', 'V2VfQ0RKb0JMZW14WHRBTENhYWI6MW93UjJrZ2s1ZEVWcXdUdW1CVENEUQ==')
ES_INDEX = os.environ.get('ES_INDEX', 'mb_blends_v1')

# Global state for lazy initialization
es_client = None
bq_client = None
gcs_client = None

# Mock motion data
MOCK_MOTIONS = [
    {
        "id": "seed_walk_001",
        "name": "Walking Forward",
        "category": "seed",
        "metadata": {
            "duration": 2.5,
            "frames": 75,
            "fps": 30,
            "joints": 24,
            "file_format": "BVH",
            "tags": ["locomotion", "basic", "forward"],
            "connectable_to": ["seed_walk_002", "seed_run_001", "seed_jump_001"]
        }
    },
    {
        "id": "seed_walk_002",
        "name": "Walking Backward",
        "category": "seed",
        "metadata": {
            "duration": 2.5,
            "frames": 75,
            "fps": 30,
            "joints": 24,
            "file_format": "BVH",
            "tags": ["locomotion", "basic", "backward"],
            "connectable_to": ["seed_walk_001", "seed_run_002", "seed_idle_001"]
        }
    },
    {
        "id": "seed_run_001",
        "name": "Running Sprint",
        "category": "seed",
        "metadata": {
            "duration": 1.8,
            "frames": 54,
            "fps": 30,
            "joints": 24,
            "file_format": "BVH",
            "tags": ["locomotion", "fast", "sprint"],
            "connectable_to": ["seed_walk_001", "seed_jump_001", "seed_run_002"]
        }
    },
    {
        "id": "seed_run_002",
        "name": "Jogging",
        "category": "seed",
        "metadata": {
            "duration": 3.0,
            "frames": 90,
            "fps": 30,
            "joints": 24,
            "file_format": "BVH",
            "tags": ["locomotion", "moderate", "jog"],
            "connectable_to": ["seed_walk_002", "seed_run_001", "seed_idle_001"]
        }
    },
    {
        "id": "seed_jump_001",
        "name": "Jump Landing",
        "category": "seed",
        "metadata": {
            "duration": 1.2,
            "frames": 36,
            "fps": 30,
            "joints": 24,
            "file_format": "BVH",
            "tags": ["action", "vertical", "landing"],
            "connectable_to": ["seed_walk_001", "seed_run_001", "seed_idle_001"]
        }
    },
    {
        "id": "seed_jump_002",
        "name": "High Jump",
        "category": "seed",
        "metadata": {
            "duration": 1.5,
            "frames": 45,
            "fps": 30,
            "joints": 24,
            "file_format": "BVH",
            "tags": ["action", "vertical", "high"],
            "connectable_to": ["seed_jump_001", "seed_run_001", "seed_idle_001"]
        }
    },
    {
        "id": "seed_idle_001",
        "name": "Standing Idle",
        "category": "seed",
        "metadata": {
            "duration": 4.0,
            "frames": 120,
            "fps": 30,
            "joints": 24,
            "file_format": "BVH",
            "tags": ["pose", "static", "idle"],
            "connectable_to": ["seed_walk_001", "seed_walk_002", "seed_jump_001"]
        }
    },
    {
        "id": "seed_idle_002",
        "name": "Crouching",
        "category": "seed",
        "metadata": {
            "duration": 2.0,
            "frames": 60,
            "fps": 30,
            "joints": 24,
            "file_format": "BVH",
            "tags": ["pose", "static", "crouch"],
            "connectable_to": ["seed_idle_001", "seed_walk_001", "seed_jump_002"]
        }
    }
]

# In-memory artifacts store (replace with database in production)
ARTIFACTS_STORE = [
    {
        "id": "blend_walkrun_001",
        "name": "Walking to Running Transition",
        "type": "blend",
        "status": "completed",
        "created_at": "2025-10-22T10:00:00Z",
        "metadata": {
            "source_motions": ["seed_walk_001", "seed_run_001"],
            "blend_weight": 0.6,
            "frames": 95,
            "duration": 3.17,
            "quality_score": 0.87
        }
    },
    {
        "id": "blend_jumpidle_001",
        "name": "Jump to Idle Pose",
        "type": "blend",
        "status": "completed",
        "created_at": "2025-10-22T11:15:00Z",
        "metadata": {
            "source_motions": ["seed_jump_001", "seed_idle_001"],
            "blend_weight": 0.4,
            "frames": 78,
            "duration": 2.6,
            "quality_score": 0.82
        }
    },
    {
        "id": "blend_runjump_001",
        "name": "Running to Jump",
        "type": "blend",
        "status": "completed",
        "created_at": "2025-10-22T12:30:00Z",
        "metadata": {
            "source_motions": ["seed_run_001", "seed_jump_002"],
            "blend_weight": 0.7,
            "frames": 67,
            "duration": 2.23,
            "quality_score": 0.91
        }
    }
]


def get_gcs_client():
    """Lazy initialization of GCS client"""
    global gcs_client
    if gcs_client is None:
        try:
            from google.cloud import storage
            gcs_client = storage.Client(project=BQ_PROJECT)
            logger.info("✅ GCS client initialized")
        except Exception as e:
            logger.warning(f"⚠️ GCS client unavailable: {e}")
    return gcs_client


def get_bigquery_client():
    """Lazy initialization of BigQuery client"""
    global bq_client
    if bq_client is None:
        try:
            from google.cloud import bigquery
            bq_client = bigquery.Client(project=BQ_PROJECT)
            logger.info("✅ BigQuery client initialized")
        except Exception as e:
            logger.warning(f"⚠️ BigQuery client unavailable: {e}")
    return bq_client


def get_elasticsearch_client():
    """Lazy initialization of Elasticsearch client"""
    global es_client
    if es_client is None:
        try:
            from elasticsearch import Elasticsearch
            # Use API key authentication for Elastic Cloud
            es_client = Elasticsearch(
                ES_URL,
                api_key=ES_API_KEY,
                verify_certs=True
            )
            es_client.info()
            logger.info("✅ Elasticsearch client initialized (Elastic Cloud)")
        except Exception as e:
            logger.warning(f"⚠️ Elasticsearch unavailable: {e}")
    return es_client


@app.route('/version', methods=['GET', 'OPTIONS'])
def version():
    """Version information endpoint"""
    if request.method == 'OPTIONS':
        return '', 204
    
    return jsonify({
        "api_version": "1.0.0",
        "ui_version": "1.0.0",
        "build_date": "2025-10-22",
        "git_commit": "f1b6f7ab",
        "environment": "production",
        "server": "moverse.rydlr.com",
        "services": {
            "elasticsearch": ES_URL,
            "bigquery_project": BQ_PROJECT,
            "gcs_bucket": GCS_BUCKET
        }
    })


# ============================================================================
# HEALTH & STATUS ENDPOINTS
# ============================================================================

@app.route('/health', methods=['GET', 'OPTIONS'])
def health():
    """Health check endpoint"""
    if request.method == 'OPTIONS':
        return '', 204
    
    return jsonify({
        "status": "healthy",
        "service": "MotionBlend AI API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    })


@app.route('/status', methods=['GET', 'OPTIONS'])
def status():
    """Detailed status endpoint"""
    if request.method == 'OPTIONS':
        return '', 204
    
    return jsonify({
        "service": "MotionBlend AI API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "gcs": gcs_client is not None,
            "bigquery": bq_client is not None,
            "elasticsearch": es_client is not None
        },
        "config": {
            "gcs_bucket": GCS_BUCKET,
            "bq_project": BQ_PROJECT,
            "bq_dataset": BQ_DATASET,
            "es_index": ES_INDEX
        }
    })


# ============================================================================
# MOTION LIBRARY ENDPOINTS
# ============================================================================

@app.route('/motions', methods=['GET', 'OPTIONS'])
def get_motions():
    """
    Get list of available motion capture files
    
    Query params:
    - category: Filter by category (seed, build, blend)
    - limit: Max number of results (default: 100)
    """
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        category = request.args.get('category')
        limit = int(request.args.get('limit', 100))
        
        # Try to get from BigQuery
        bq = get_bigquery_client()
        if bq:
            try:
                query = f"""
                SELECT 
                    motion_id as id,
                    motion_name as name,
                    category,
                    frame_count as frames,
                    created_at,
                    updated_at
                FROM `{BQ_PROJECT}.{BQ_DATASET}.seed_motions`
                """
                
                if category:
                    query += f" WHERE category = '{category}'"
                
                query += f" LIMIT {limit}"
                
                results = bq.query(query).result()
                motions = []
                
                for row in results:
                    motions.append({
                        "id": row.id,
                        "name": row.name,
                        "category": row.category,
                        "metadata": {
                            "frames": row.frames,
                            "created_at": row.created_at.isoformat() if row.created_at else None,
                            "updated_at": row.updated_at.isoformat() if row.updated_at else None
                        }
                    })
                
                return jsonify({"motions": motions, "total": len(motions)})
                
            except Exception as e:
                logger.warning(f"BigQuery query failed: {e}")
        
        # Fallback to mock data
        motions = MOCK_MOTIONS
        if category:
            motions = [m for m in motions if m.get('category') == category]
        
        return jsonify({"motions": motions[:limit], "total": len(motions)})
        
    except Exception as e:
        logger.error(f"Error fetching motions: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# BLEND OPERATION ENDPOINTS
# ============================================================================

@app.route('/api/blend', methods=['POST', 'OPTIONS'])
def create_blend():
    """
    Create a new motion blend
    
    Request body:
    {
        "motion1": "walking_forward",
        "motion2": "jump_landing",
        "weight": 0.5
    }
    """
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.get_json()
        motion1 = data.get('motion1')
        motion2 = data.get('motion2')
        weight = data.get('weight', 0.5)
        
        if not motion1 or not motion2:
            return jsonify({"error": "Both motion1 and motion2 are required"}), 400
        
        if not 0.0 <= weight <= 1.0:
            return jsonify({"error": "Weight must be between 0.0 and 1.0"}), 400
        
        # Generate blend ID
        blend_id = hashlib.sha1(
            f"{motion1}_{motion2}_{weight}_{datetime.now().timestamp()}".encode()
        ).hexdigest()[:12]
        
        # Create blend artifact
        artifact = {
            "id": f"blend_{blend_id}",
            "name": f"{motion1}_{motion2}_blend",
            "type": "blend",
            "status": "completed",
            "created_at": datetime.now().isoformat(),
            "metadata": {
                "source_motions": [motion1, motion2],
                "blend_weight": weight,
                "frames": 120,  # Estimated
                "duration": 4.0,  # Estimated
                "quality_score": 0.85  # Mock quality score
            }
        }
        
        # Store artifact
        ARTIFACTS_STORE.append(artifact)
        
        logger.info(f"✅ Blend created: {artifact['name']}")
        
        return jsonify(artifact), 201
        
    except Exception as e:
        logger.error(f"Blend creation error: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# ARTIFACTS ENDPOINTS
# ============================================================================

@app.route('/api/artifacts', methods=['GET', 'OPTIONS'])
def get_artifacts():
    """Get all artifacts"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        limit = int(request.args.get('limit', 100))
        
        # Try to get from BigQuery
        bq = get_bigquery_client()
        if bq:
            try:
                query = f"""
                SELECT 
                    blend_id as id,
                    motion1_name || '_' || motion2_name || '_blend' as name,
                    created_at,
                    quality_score
                FROM `{BQ_PROJECT}.{BQ_DATASET}_marts.mart_blend_snn_metrics`
                ORDER BY created_at DESC
                LIMIT {limit}
                """
                
                results = bq.query(query).result()
                artifacts = []
                
                for row in results:
                    artifacts.append({
                        "id": row.id,
                        "name": row.name,
                        "created_at": row.created_at.isoformat() if row.created_at else None,
                        "quality_score": float(row.quality_score) if row.quality_score else None
                    })
                
                return jsonify({"artifacts": artifacts, "total": len(artifacts)})
                
            except Exception as e:
                logger.warning(f"BigQuery query failed: {e}")
        
        # Fallback to in-memory store
        return jsonify({
            "artifacts": ARTIFACTS_STORE[:limit],
            "total": len(ARTIFACTS_STORE)
        })
        
    except Exception as e:
        logger.error(f"Error fetching artifacts: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/artifacts/manifest', methods=['GET', 'OPTIONS'])
def get_artifacts_manifest():
    """Get artifacts manifest with metadata"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        # Try to get from GCS
        gcs = get_gcs_client()
        if gcs:
            try:
                bucket = gcs.bucket(GCS_BUCKET)
                blobs = bucket.list_blobs(prefix='blends/', max_results=100)
                
                artifacts = []
                for blob in blobs:
                    if blob.name.endswith('.bvh'):
                        artifacts.append({
                            "id": blob.name.split('/')[-1].replace('.bvh', ''),
                            "name": blob.name.split('/')[-1],
                            "created_at": blob.time_created.isoformat(),
                            "size": blob.size,
                            "metadata": blob.metadata or {}
                        })
                
                return jsonify({
                    "artifacts": artifacts,
                    "total": len(artifacts),
                    "last_updated": datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.warning(f"GCS query failed: {e}")
        
        # Fallback to in-memory store
        return jsonify({
            "artifacts": ARTIFACTS_STORE,
            "total": len(ARTIFACTS_STORE),
            "last_updated": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error fetching manifest: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/artifact/<artifact_id>/describe', methods=['GET', 'OPTIONS'])
def describe_artifact(artifact_id):
    """Get detailed artifact description"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        # Try to get from BigQuery
        bq = get_bigquery_client()
        if bq:
            try:
                query = f"""
                SELECT 
                    blend_id,
                    motion1_name,
                    motion2_name,
                    blend_weight,
                    quality_score,
                    quality_category,
                    created_at
                FROM `{BQ_PROJECT}.{BQ_DATASET}_marts.mart_blend_snn_metrics`
                WHERE blend_id = '{artifact_id}'
                LIMIT 1
                """
                
                results = bq.query(query).result()
                
                for row in results:
                    return jsonify({
                        "id": row.blend_id,
                        "motion1": row.motion1_name,
                        "motion2": row.motion2_name,
                        "weight": float(row.blend_weight) if row.blend_weight else None,
                        "quality_score": float(row.quality_score) if row.quality_score else None,
                        "quality_category": row.quality_category,
                        "created_at": row.created_at.isoformat() if row.created_at else None
                    })
                
                return jsonify({"error": "Artifact not found"}), 404
                
            except Exception as e:
                logger.warning(f"BigQuery query failed: {e}")
        
        # Fallback to in-memory store
        for artifact in ARTIFACTS_STORE:
            if artifact['id'] == artifact_id:
                return jsonify(artifact)
        
        return jsonify({"error": "Artifact not found"}), 404
        
    except Exception as e:
        logger.error(f"Error describing artifact: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/artifact/<artifact_id>/analysis', methods=['GET', 'OPTIONS'])
def get_artifact_analysis(artifact_id):
    """Get artifact quality metrics and analysis"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        # Try to get from BigQuery
        bq = get_bigquery_client()
        if bq:
            try:
                query = f"""
                SELECT 
                    blend_id,
                    quality_score,
                    quality_category,
                    fid,
                    coverage,
                    global_diversity,
                    local_diversity,
                    l2_velocity_mean,
                    l2_acceleration_mean,
                    transition_smoothness,
                    has_velocity_spike,
                    has_rough_transition,
                    has_distribution_mismatch
                FROM `{BQ_PROJECT}.{BQ_DATASET}_marts.mart_blend_snn_metrics`
                WHERE blend_id = '{artifact_id}'
                LIMIT 1
                """
                
                results = bq.query(query).result()
                
                for row in results:
                    return jsonify({
                        "id": row.blend_id,
                        "quality": {
                            "score": float(row.quality_score) if row.quality_score else None,
                            "category": row.quality_category
                        },
                        "metrics": {
                            "fid": float(row.fid) if row.fid else None,
                            "coverage": float(row.coverage) if row.coverage else None,
                            "global_diversity": float(row.global_diversity) if row.global_diversity else None,
                            "local_diversity": float(row.local_diversity) if row.local_diversity else None,
                            "l2_velocity_mean": float(row.l2_velocity_mean) if row.l2_velocity_mean else None,
                            "l2_acceleration_mean": float(row.l2_acceleration_mean) if row.l2_acceleration_mean else None,
                            "transition_smoothness": float(row.transition_smoothness) if row.transition_smoothness else None
                        },
                        "issues": {
                            "velocity_spike": bool(row.has_velocity_spike),
                            "rough_transition": bool(row.has_rough_transition),
                            "distribution_mismatch": bool(row.has_distribution_mismatch)
                        }
                    })
                
                return jsonify({"error": "Analysis not found"}), 404
                
            except Exception as e:
                logger.warning(f"BigQuery query failed: {e}")
        
        # Fallback to mock data
        # Generate varied metrics based on artifact_id for demonstration
        import random
        random.seed(hash(artifact_id) % 1000)  # Deterministic but varied
        
        quality_score = round(random.uniform(0.6, 0.95), 2)
        if quality_score >= 0.85:
            category = "excellent"
        elif quality_score >= 0.75:
            category = "good"
        elif quality_score >= 0.65:
            category = "fair"
        else:
            category = "poor"
        
        return jsonify({
            "id": artifact_id,
            "quality": {
                "score": quality_score,
                "category": category
            },
            "metrics": {
                "fid": round(random.uniform(8.0, 25.0), 1),
                "coverage": round(random.uniform(0.5, 0.9), 2),
                "global_diversity": round(random.uniform(0.4, 0.8), 2),
                "local_diversity": round(random.uniform(0.3, 0.7), 2),
                "l2_velocity_mean": round(random.uniform(0.8, 2.5), 1),
                "l2_acceleration_mean": round(random.uniform(0.5, 1.8), 1),
                "transition_smoothness": round(random.uniform(0.6, 0.9), 2),
                "joint_coherence": round(random.uniform(0.7, 0.95), 2),
                "temporal_consistency": round(random.uniform(0.75, 0.95), 2),
                "pose_realism": round(random.uniform(0.6, 0.9), 2)
            },
            "issues": {
                "velocity_spike": random.choice([True, False]),
                "rough_transition": random.choice([True, False]),
                "distribution_mismatch": random.choice([True, False]),
                "joint_discontinuity": random.choice([True, False]),
                "temporal_artifacts": random.choice([True, False])
            },
            "recommendations": [
                "Consider adjusting blend weights for smoother transitions" if random.random() > 0.5 else None,
                "Increase training data diversity" if quality_score < 0.8 else None,
                "Fine-tune temporal alignment" if random.random() > 0.7 else None
            ]
        })
        
    except Exception as e:
        logger.error(f"Error fetching analysis: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# SEARCH ENDPOINTS (Elasticsearch)
# ============================================================================

@app.route('/search/vector', methods=['POST', 'OPTIONS'])
def search_vector():
    """
    Vector similarity search
    
    Request body:
    {
        "vector": [0.1, 0.2, ...],  // 384-dim embedding
        "k": 5
    }
    """
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.get_json()
        query_vector = data.get('vector', [])
        k = data.get('k', 5)
        
        if not query_vector:
            return jsonify({"error": "Vector is required"}), 400
        
        # Try Elasticsearch
        es = get_elasticsearch_client()
        if es:
            try:
                search_body = {
                    "size": k,
                    "query": {
                        "knn": {
                            "field": "motion_vector",
                            "query_vector": query_vector,
                            "k": k,
                            "num_candidates": k * 2
                        }
                    }
                }
                
                response = es.search(index=ES_INDEX, body=search_body)
                
                results = []
                for hit in response['hits']['hits']:
                    result = hit['_source']
                    result['score'] = hit['_score']
                    results.append(result)
                
                return jsonify({"results": results, "total": len(results)})
                
            except Exception as e:
                logger.warning(f"Elasticsearch search failed: {e}")
        
        # Fallback to mock results
        return jsonify({
            "results": [],
            "total": 0,
            "note": "Elasticsearch not available, no results"
        })
        
    except Exception as e:
        logger.error(f"Vector search error: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# STATIC FILES (Optional - for serving UI)
# ============================================================================

@app.route('/', methods=['GET'])
def index():
    """Serve API documentation or redirect to UI"""
    return jsonify({
        "service": "MotionBlend AI API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "status": "/status",
            "motions": "/motions",
            "blend": "/api/blend",
            "artifacts": "/api/artifacts",
            "manifest": "/api/artifacts/manifest",
            "describe": "/api/artifact/{id}/describe",
            "analysis": "/api/artifact/{id}/analysis",
            "search": "/search/vector"
        },
        "documentation": "https://github.com/RydlrCS/MotionBlendAI"
    })


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("🚀 MotionBlend AI Production API Server")
    print("=" * 60)
    print(f"📊 Environment:")
    print(f"   GCS Bucket: {GCS_BUCKET}")
    print(f"   BigQuery Project: {BQ_PROJECT}")
    print(f"   BigQuery Dataset: {BQ_DATASET}")
    print(f"   Elasticsearch URL: {ES_URL}")
    print(f"   Elasticsearch Index: {ES_INDEX}")
    print("=" * 60)
    print(f"🌐 Server starting on http://0.0.0.0:8080")
    print("✅ CORS enabled for all origins")
    print("⚡ Lazy initialization: Services connect on first request")
    print("=" * 60)
    
    # Production settings
    app.run(
        debug=False,  # Set to True for development
        host='0.0.0.0',  # Listen on all interfaces
        port=8080,
        threaded=True
    )
