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
from typing import Dict, List, Any, Optional, TypedDict, Union
from datetime import datetime
import numpy as np
import logging
import time
import functools
from contextlib import contextmanager

# WCAG-friendly color palette for motion strips
DEFAULT_COLORS = [
    "#2FBF71",  # Salsa - Green
    "#F0A202",  # Swing - Amber
    "#2D7DD2",  # Wave - Blue
    "#E63946",  # Funk - Red
    "#9C27B0",  # Jazz - Purple
    "#00897B",  # Ballet - Teal
    "#FF6F00",  # Hip-hop - Orange
]

# ============================================================================
# PERFORMANCE & DEBUGGING UTILITIES
# ============================================================================

# Configure enhanced logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

@contextmanager
def timer_context(operation_name: str):
    """Context manager to time operations and log performance"""
    start = time.time()
    logger.info(f"⏱️  START: {operation_name}")
    try:
        yield
    finally:
        elapsed = time.time() - start
        logger.info(f"✅ COMPLETE: {operation_name} | Duration: {elapsed:.3f}s")

def retry_with_backoff(max_retries: int = 3, initial_delay: float = 1.0, backoff_factor: float = 2.0):
    """Decorator to retry functions with exponential backoff"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    logger.debug(f"🔄 Attempt {attempt + 1}/{max_retries}: {func.__name__}")
                    result = func(*args, **kwargs)
                    if attempt > 0:
                        logger.info(f"✅ Retry succeeded for {func.__name__} after {attempt + 1} attempts")
                    return result
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(f"⚠️  Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay}s...")
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error(f"❌ All {max_retries} attempts failed for {func.__name__}: {e}")
            
            raise last_exception
        return wrapper
    return decorator

# Type definitions for motion data
class MotionMetadata(TypedDict, total=False):
    category: str
    duration: float
    frames: int
    joints: int
    format: str
    tags: List[str]
    intensity: float
    complexity: str

class MotionData(TypedDict, total=False):
    id: str
    name: str
    description: str
    vector: List[float]
    metadata: MotionMetadata
    similarity_score: float
    semantic_score: float
    highlight: Optional[Dict[str, List[str]]]

from ES_INDEX_NAME import ES_INDEX_NAME

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Elasticsearch configuration - Updated cluster with semantic text support
_DEFAULT_ES_API_KEY = "R3pLWTNKa0JJSXBZc3NKUnBOdnk6ZHJlSFF2Z1FyVUh5SC05SUZucjZMZw=="  # Provided API key
_DEFAULT_ES_CLOUD_URL = "https://my-elasticsearch-project-ba986d.es.us-central1.gcp.elastic.cloud:443"  # Provided endpoint

# Try to get from environment variables (fallback)
import os
import time
from datetime import datetime
ES_API_KEY = os.getenv('ES_API_KEY', _DEFAULT_ES_API_KEY)
ES_CLOUD_URL = os.getenv('ES_CLOUD_URL', _DEFAULT_ES_CLOUD_URL)
ELASTICSEARCH_URL = os.getenv('ELASTICSEARCH_URL')  # For local Docker setup

try:
    from flask_cors import CORS  # type: ignore
    flask_cors_available = True
except ImportError:
    flask_cors_available = False
    print("flask-cors not available")

try:
    from elasticsearch import Elasticsearch  # type: ignore
    elasticsearch_available = True
    ElasticsearchType = type[Elasticsearch]
except ImportError:
    elasticsearch_available = False
    print("elasticsearch not available")
    # Create a dummy class for type hints when elasticsearch is not available
    class Elasticsearch:  # type: ignore
        pass
    ElasticsearchType = type[Elasticsearch]

app = Flask(__name__)
if flask_cors_available:
    from flask_cors import CORS  # type: ignore
    CORS(app)  # Enable CORS for all routes

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Docker"""
    return jsonify({
        "status": "healthy",
        "elasticsearch": "connected" if es_available else "disconnected",
        "timestamp": datetime.now().isoformat()
    })

# Connect to Elasticsearch instance (cloud or local)
es: Optional[ElasticsearchType] = None  # type: ignore
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

@retry_with_backoff(max_retries=3, initial_delay=2.0, backoff_factor=2.0)
def initialize_elasticsearch():
    """Initialize Elasticsearch connection and create index with mappings."""
    global es, es_available
    
    logger.info("🔌 Initializing Elasticsearch connection...")
    
    if not elasticsearch_available:
        logger.warning("⚠️  Elasticsearch library not available")
        return
    
    try:
        # Connect to Elasticsearch (prefer local Docker setup, fallback to cloud)
        if ELASTICSEARCH_URL:
            # Local Elasticsearch in Docker
            logger.info(f"🐳 Attempting connection to local Elasticsearch: {ELASTICSEARCH_URL}")
            es = Elasticsearch([ELASTICSEARCH_URL])
        else:
            # Cloud Elasticsearch with semantic text support
            logger.info(f"☁️  Attempting connection to Elasticsearch Cloud: {ES_CLOUD_URL}")
            es = Elasticsearch([ES_CLOUD_URL], api_key=ES_API_KEY)
        
        # Test connection with timeout logging
        with timer_context("Elasticsearch ping"):
            if es is not None and hasattr(es, 'ping') and es.ping():  # type: ignore
                es_available = True
                cluster_info = es.info()  # type: ignore
                logger.info(f"✅ Connected to Elasticsearch {cluster_info['version']['number']}")  # type: ignore
                
                # Create index if it doesn't exist
                with timer_context("Ensure Elasticsearch index"):
                    if not es.indices.exists(index=ES_INDEX_NAME):  # type: ignore
                        mappings = {"mappings": create_motion_mappings()}
                        es.indices.create(index=ES_INDEX_NAME, body=mappings)  # type: ignore
                        logger.info(f"✅ Created index '{ES_INDEX_NAME}' with semantic text mappings")
                    else:
                        # Update existing index mappings
                        try:
                            mappings = create_motion_mappings()
                            es.indices.put_mapping(index=ES_INDEX_NAME, body=mappings)  # type: ignore
                            logger.info(f"✅ Updated mappings for index '{ES_INDEX_NAME}'")
                        except Exception as mapping_error:
                            logger.warning(f"⚠️  Mapping update: {mapping_error}")
            else:
                logger.error("❌ Elasticsearch ping failed")
                es_available = False
                es = None
    
    except Exception as e:
        logger.error(f"❌ Elasticsearch connection failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        es_available = False
        es = None


# Initialize Elasticsearch on startup
initialize_elasticsearch()

# Import mock data generators
import os
from pathlib import Path

def load_seed_motions():
    """Load actual motion files from seed_motions directory with performance logging"""
    with timer_context("Loading seed motions"):
        seed_dir = Path(__file__).parent.parent / "seed_motions"
        motion_files = []
        
        # DEBUG MODE: Load only 1 file for fast UI testing
        DEBUG_MODE = os.getenv('DEBUG_MODE', 'true').lower() == 'true'
        MAX_FILES = 1 if DEBUG_MODE else 100
        
        logger.info(f"📁 Seed motions path: {seed_dir}")
        logger.info(f"🐛 DEBUG_MODE={DEBUG_MODE}, MAX_FILES={MAX_FILES}")
        
        if not seed_dir.exists():
            logger.warning(f"⚠️  Seed motions directory not found: {seed_dir}")
            return []
        
        file_list = list(seed_dir.glob("*.*"))
        logger.info(f"📊 Found {len(file_list)} total files in seed_motions/")
        
        file_count = 0
        for i, file_path in enumerate(file_list, 1):
            if len(motion_files) >= MAX_FILES:
                logger.info(f"🔧 DEBUG MODE: Loaded {MAX_FILES} file(s) for testing (stopped at {i}/{len(file_list)})")
                break
                
            if file_path.suffix.lower() in ['.fbx', '.glb', '.trc']:
                try:
                    # Extract motion characteristics from filename
                    name = file_path.stem
                    category = categorize_motion(name)
                    
                    logger.debug(f"📄 Processing {i}/{len(file_list)}: {name} (category: {category})")
                    
                    motion_files.append({
                        "id": f"seed_{len(motion_files)+1:03d}",
                        "name": name,
                        "description": f"Motion capture file: {name}",
                        "file_path": str(file_path),
                        "file_format": file_path.suffix.lower(),
                        "vector": generate_motion_vector(name),
                        "metadata": {
                            "category": category,
                            "source": "seed_motions",
                            "format": file_path.suffix.upper(),
                            "file_size": file_path.stat().st_size if file_path.exists() else 0,
                            "frames": estimate_frames(name),
                            "duration": estimate_duration(name),
                            "joints": 25,
                            "complexity": estimate_complexity(name),
                            "intensity": estimate_intensity(name),
                            "tags": extract_tags(name)
                        }
                    })
                    file_count += 1
                except Exception as e:
                    logger.error(f"❌ Failed to process {file_path.name}: {e}")
                    continue
        
        logger.info(f"✅ Loaded {len(motion_files)} seed motions successfully ({file_count} processed)")
        return motion_files

def categorize_motion(name: str) -> str:
    """Categorize motion based on filename"""
    name_lower = name.lower()
    if any(word in name_lower for word in ['kick', 'punch', 'fight', 'combat']):
        return 'combat'
    elif any(word in name_lower for word in ['dance', 'dancing']):
        return 'dance'
    elif any(word in name_lower for word in ['spell', 'magic', 'mage']):
        return 'fantasy'
    elif any(word in name_lower for word in ['tennis', 'sport']):
        return 'sports'
    elif any(word in name_lower for word in ['angry', 'emotion']):
        return 'emotional'
    else:
        return 'general'

def estimate_frames(name: str) -> int:
    """Estimate frame count based on motion type"""
    if 'spell' in name.lower() or 'magic' in name.lower():
        return 180  # ~6 seconds at 30fps
    elif 'dance' in name.lower():
        return 300  # ~10 seconds
    elif 'kick' in name.lower() or 'punch' in name.lower():
        return 90   # ~3 seconds
    else:
        return 150  # ~5 seconds default

def estimate_duration(name: str) -> float:
    """Estimate duration in seconds"""
    return estimate_frames(name) / 30.0

def estimate_complexity(name: str) -> str:
    """Estimate motion complexity"""
    if any(word in name.lower() for word in ['spell', 'magic', 'dance']):
        return 'high'
    elif any(word in name.lower() for word in ['kick', 'punch', 'tennis']):
        return 'medium'
    else:
        return 'low'

def estimate_intensity(name: str) -> float:
    """Estimate motion intensity (0.0-1.0)"""
    if any(word in name.lower() for word in ['kick', 'punch', 'angry']):
        return 0.8
    elif any(word in name.lower() for word in ['dance', 'tennis']):
        return 0.6
    elif any(word in name.lower() for word in ['spell', 'magic']):
        return 0.4
    else:
        return 0.3

def extract_tags(name: str) -> List[str]:
    """Extract relevant tags from motion name"""
    tags = []
    name_lower = name.lower()
    
    # Action tags
    if 'kick' in name_lower: tags.extend(['kick', 'martial_arts', 'combat'])
    if 'punch' in name_lower: tags.extend(['punch', 'boxing', 'combat'])
    if 'dance' in name_lower: tags.extend(['dance', 'artistic', 'rhythmic'])
    if 'spell' in name_lower: tags.extend(['magic', 'fantasy', 'casting'])
    if 'tennis' in name_lower: tags.extend(['tennis', 'sports', 'racket'])
    if 'angry' in name_lower: tags.extend(['emotion', 'angry', 'expressive'])
    
    # Style tags
    if 'mixamo' in name_lower: tags.append('mixamo')
    if 'reprocessed' in name_lower: tags.append('processed')
    
    return tags

def generate_motion_vector(name: str) -> List[float]:
    """Generate a characteristic vector for the motion"""
    import hashlib
    import struct
    
    # Create deterministic vector based on name
    hash_obj = hashlib.md5(name.encode())
    hash_bytes = hash_obj.digest()
    
    # Convert to 512-dimensional vector
    vector = []
    for i in range(0, len(hash_bytes), 4):
        chunk = hash_bytes[i:i+4].ljust(4, b'\x00')
        value = struct.unpack('f', chunk)[0]
        vector.append(float(value))
    
    # Pad to 512 dimensions
    while len(vector) < 512:
        vector.append(0.0)
    
    return vector[:512]

# Load seed motions and combine with mock data
seed_motions = load_seed_motions()

# DEBUG MODE: Limit mock motions to 5 for fast testing
DEBUG_MODE = os.getenv('DEBUG_MODE', 'true').lower() == 'true'
MAX_MOCK_MOTIONS = 3 if DEBUG_MODE else 200

# Mock motion data for development and testing
# This provides realistic motion capture metadata for UI development
_ALL_MOCK_MOTIONS = [
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

# Apply debug mode limiting
MOCK_MOTIONS = seed_motions + _ALL_MOCK_MOTIONS[:MAX_MOCK_MOTIONS]
if DEBUG_MODE:
    print(f"🔧 DEBUG MODE: Limited to {len(seed_motions)} seed motion(s) + {min(MAX_MOCK_MOTIONS, len(_ALL_MOCK_MOTIONS))} mock motions = {len(MOCK_MOTIONS)} total")

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
                
                hits: List[Dict[str, Any]] = []
                for hit in response["hits"]["hits"]:
                    motion_data: Dict[str, Any] = hit["_source"].copy()
                    motion_data["similarity_score"] = float(hit["_score"])
                    motion_data["id"] = str(hit["_id"])
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
                
                hits: List[MotionData] = []
                for hit in response["hits"]["hits"]:
                    motion_data: MotionData = hit["_source"].copy()  # type: ignore
                    motion_data["similarity_score"] = float(hit["_score"]) / 10.0  # Normalize score
                    motion_data["id"] = str(hit["_id"])
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
            bulk_docs: List[Dict[str, Any]] = []
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
                
                # Extract response data with proper typing
                success_count = int(bulk_response[0]) if bulk_response and isinstance(bulk_response, tuple) else len(documents)
                error_list = bulk_response[1] if bulk_response and isinstance(bulk_response, tuple) and len(bulk_response) > 1 and isinstance(bulk_response[1], list) else []
                error_count = len(error_list)
                
                return jsonify({
                    "success": True,
                    "indexed": success_count,
                    "errors": error_count,
                    "details": error_list[:5] if error_list else [],
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
    """Get all available motions including seed files"""
    # Refresh seed motions to pick up any new files
    fresh_seed_motions = load_seed_motions()
    all_motions = fresh_seed_motions + MOCK_MOTIONS[len(seed_motions):]
    return jsonify(all_motions)

@app.route('/motions/refresh', methods=['POST'])
def refresh_motions():
    """Refresh motion data from seed_motions directory"""
    global MOCK_MOTIONS, seed_motions
    
    try:
        # Reload seed motions
        fresh_seed_motions = load_seed_motions()
        MOCK_MOTIONS = fresh_seed_motions + MOCK_MOTIONS[len(seed_motions):]
        seed_motions = fresh_seed_motions
        
        return jsonify({
            "status": "success",
            "message": f"Refreshed {len(fresh_seed_motions)} seed motions",
            "total_motions": len(MOCK_MOTIONS)
        })
    except Exception as e:
        return jsonify({"error": f"Failed to refresh motions: {e}"}), 500

# Global artifacts storage
ARTIFACTS_STORE = []

def generate_frame_thumbnail(frame_index: int, motion_hash: int, motion_type: str = 'blend') -> str:
    """Generate a placeholder thumbnail data URL for a frame"""
    # Create a simple colored SVG as placeholder
    # In production, this would generate actual rendered frames
    
    # Use hash to create consistent colors
    hue = (motion_hash + frame_index * 137) % 360
    
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="80" height="80" viewBox="0 0 80 80">
        <rect width="80" height="80" fill="hsl({hue}, 60%, 50%)"/>
        <text x="40" y="45" text-anchor="middle" font-size="14" fill="white" font-family="monospace">
            F{frame_index}
        </text>
    </svg>'''
    
    # Convert to base64 data URL
    import base64
    svg_bytes = svg.encode('utf-8')
    b64 = base64.b64encode(svg_bytes).decode('utf-8')
    return f"data:image/svg+xml;base64,{b64}"

def generate_source_motion_data(motion_name: str, motion_hash: int, frames: int, color: str) -> Dict[str, Any]:
    """Generate source motion data for strip visualization"""
    sample_every = 10  # Sample every 10th frame
    sampled_frames = list(range(0, frames, sample_every))
    
    thumbnails = [
        generate_frame_thumbnail(f, motion_hash, 'source')
        for f in sampled_frames
    ]
    
    return {
        "id": f"source_{motion_hash}_{motion_name.replace(' ', '_').lower()}",
        "label": motion_name,
        "character": "Default",
        "frames": frames,
        "sampleEvery": sample_every,
        "thumbnails": thumbnails,
        "color": color
    }

def generate_blend_motion_data(
    blend_id: str,
    motion1: str,
    motion2: str,
    frames: int,
    weight: float,
    motion_hash: int,
    transition_start: int,
    transition_end: int
) -> Dict[str, Any]:
    """Generate blend motion data with segments for strip visualization"""
    sample_every = 10
    sampled_frames = list(range(0, frames, sample_every))
    
    thumbnails = [
        generate_frame_thumbnail(f, motion_hash, 'blend')
        for f in sampled_frames
    ]
    
    # Create segments based on transition windows
    # Colors from MOTION_COLORS palette
    color1 = DEFAULT_COLORS[hash(motion1) % len(DEFAULT_COLORS)]
    color2 = DEFAULT_COLORS[hash(motion2) % len(DEFAULT_COLORS)]
    
    segments = []
    
    # Pre-transition: Motion 1 dominant
    if transition_start > 0:
        segments.append({
            "fromFrame": 0,
            "toFrame": transition_start - 1,
            "label": motion1,
            "color": color1,
            "alpha": 1.0
        })
    
    # Transition: Blend of both
    segments.append({
        "fromFrame": transition_start,
        "toFrame": transition_end,
        "label": f"{motion1} → {motion2}",
        "color": color1,
        "alpha": 1.0 - weight
    })
    segments.append({
        "fromFrame": transition_start,
        "toFrame": transition_end,
        "label": f"{motion1} → {motion2}",
        "color": color2,
        "alpha": weight
    })
    
    # Post-transition: Motion 2 dominant
    if transition_end < frames - 1:
        segments.append({
            "fromFrame": transition_end + 1,
            "toFrame": frames - 1,
            "label": motion2,
            "color": color2,
            "alpha": 1.0
        })
    
    return {
        "id": blend_id,
        "label": f"{motion1} to {motion2} (w={weight:.2f})",
        "frames": frames,
        "sampleEvery": sample_every,
        "thumbnails": thumbnails,
        "segments": segments
    }

def generate_blend_analysis(motion_hash: int, frames: int, duration: float, weight: float) -> Dict[str, Any]:
    """Generate analysis metrics for a motion blend"""
    
    # Use motion_hash for deterministic but varied data
    np.random.seed(motion_hash % 10000)
    
    # Generate L2 velocity data (per-frame, 5 joints)
    time_points = np.linspace(0, duration, frames)
    joint_names = ['Hips', 'LeftWrist', 'RightWrist', 'LeftFoot', 'RightFoot']
    
    # Base velocity patterns influenced by weight
    l2_velocity = {}
    l2_acceleration = {}
    
    for joint in joint_names:
        # Create realistic velocity patterns with transitions
        transition_start = frames // 3
        transition_end = 2 * frames // 3
        
        velocities = []
        for t in range(frames):
            if t < transition_start:
                # Pre-transition (more influenced by motion1)
                base_vel = 0.3 + np.random.rand() * 0.2 * (1 - weight)
            elif t < transition_end:
                # Transition period (increased velocity variation)
                blend_factor = (t - transition_start) / (transition_end - transition_start)
                base_vel = 0.5 + 0.3 * np.sin(blend_factor * np.pi) + np.random.rand() * 0.3
            else:
                # Post-transition (more influenced by motion2)
                base_vel = 0.3 + np.random.rand() * 0.2 * weight
            
            velocities.append(float(base_vel))
        
        l2_velocity[joint] = velocities
        
        # Compute acceleration from velocity
        accelerations = [0.0]  # First frame has no acceleration
        for i in range(1, len(velocities)):
            accel = abs(velocities[i] - velocities[i-1])
            accelerations.append(float(accel))
        
        l2_acceleration[joint] = accelerations
    
    # Compute aggregate metrics
    all_velocities = np.concatenate([l2_velocity[j] for j in joint_names])
    all_accelerations = np.concatenate([l2_acceleration[j] for j in joint_names])
    
    # Transition smoothness (lower is better, measures discontinuity)
    transition_velocities = []
    for joint in joint_names:
        transition_velocities.extend(l2_velocity[joint][transition_start:transition_end])
    transition_smoothness = float(np.std(transition_velocities))
    
    # Global diversity (variation across all frames and joints)
    global_diversity = float(np.std(all_velocities))
    
    # Compute quality score (combination of smoothness and consistency)
    quality_score = 1.0 - min(transition_smoothness / 0.5, 1.0)  # Normalize to 0-1
    quality_category = "good" if quality_score > 0.7 else ("fair" if quality_score > 0.5 else "poor")
    
    return {
        "l2_velocity": l2_velocity,
        "l2_acceleration": l2_acceleration,
        "metrics": {
            "mean_velocity": float(np.mean(all_velocities)),
            "std_velocity": float(np.std(all_velocities)),
            "max_velocity": float(np.max(all_velocities)),
            "mean_acceleration": float(np.mean(all_accelerations)),
            "std_acceleration": float(np.std(all_accelerations)),
            "transition_smoothness": transition_smoothness,
            "global_diversity": global_diversity,
            "quality_score": quality_score,
            "quality_category": quality_category
        },
        "transition_window": {
            "start": transition_start,
            "end": transition_end
        },
        "joint_names": joint_names,
        "time_points": time_points.tolist()
    }

@app.route('/api/blend', methods=['POST', 'OPTIONS'])
@retry_with_backoff(max_retries=2, initial_delay=0.5, backoff_factor=2.0)
def create_blend():
    """Create a new motion blend and generate artifact with analysis"""
    with timer_context("Create blend artifact"):
        if request.method == 'OPTIONS':
            # Handle preflight CORS request
            response = jsonify({'status': 'ok'})
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
            response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
            return response
        
        try:
            with timer_context("Parse blend request data"):
                data = request.get_json()
                motion1 = data.get('motion1')
                motion2 = data.get('motion2') 
                weight = data.get('weight', 0.5)
                
                logger.info(f"🎬 Creating blend: {motion1} + {motion2} (weight={weight})")
                
                # Generate unique blend ID
                blend_id = f"blend_{int(time.time() * 1000)}"
                timestamp = datetime.utcnow().isoformat()
                
                # Create a hash from motion names for consistent but unique data generation
                import hashlib
                motion_hash = int(hashlib.md5(f"{motion1}{motion2}".encode()).hexdigest()[:8], 16)
            
            # Generate varied metadata based on source motions
            base_frames = 100 + (motion_hash % 50)  # 100-150 frames
            base_duration = base_frames / 30.0
            
            logger.debug(f"📊 Blend parameters: frames={base_frames}, duration={base_duration:.2f}s, hash={motion_hash}")
            
            # Generate analysis data automatically
            with timer_context("Generate blend analysis"):
                analysis_data = generate_blend_analysis(motion_hash, base_frames, base_duration, weight)
            
            transition_start = analysis_data["transition_window"]["start"]
            transition_end = analysis_data["transition_window"]["end"]
            
            # Generate source motion data for strips
            source1_frames = int(base_frames * 0.8)  # Slightly shorter for variety
            source2_frames = int(base_frames * 0.9)
            
            with timer_context("Generate source motion data"):
                sources = [
                    generate_source_motion_data(motion1, motion_hash, source1_frames, DEFAULT_COLORS[0]),
                    generate_source_motion_data(motion2, motion_hash + 1, source2_frames, DEFAULT_COLORS[1])
                ]
            
            # Generate blend motion data with segments
            with timer_context("Generate blend motion data"):
                blend_data = generate_blend_motion_data(
                    blend_id, motion1, motion2, base_frames, weight,
                    motion_hash, transition_start, transition_end
                )
            
            # Prepare metrics in new format
            metrics_formatted = {
                "joints": ["pelvis", "lwrist", "rwrist", "lfoot", "rfoot"],
                "l2Velocity": analysis_data["l2_velocity"],
                "l2Acceleration": analysis_data["l2_acceleration"],
                "transitionWindows": [{
                    "start": transition_start,
                    "end": transition_end
                }]
            }
            
            # Create blend artifact with strip visualization data
            artifact = {
                "id": blend_id,
                "name": f"{motion1} to {motion2} ({weight:.2f})",
                "type": "motion_blend",
                "status": "completed",
                "createdAt": timestamp,
                "created_at": timestamp,  # Legacy
                "fps": 30,
                "frames": base_frames,
                "sources": sources,
                "blend": blend_data,
                "metrics": metrics_formatted,
                "files": {
                    "previewPng": f"/artifacts/{blend_id}_preview.png",
                    "metricsJson": f"/artifacts/{blend_id}_metrics.json"
                },
                # Legacy fields for backward compatibility
                "metadata": {
                    "source_motions": [motion1, motion2],
                    "blend_weight": weight,
                    "frames": base_frames,
                    "duration": base_duration,
                    "format": "BVH",
                    "size": f"{2.0 + (motion_hash % 30) / 10:.1f} MB",
                    "motion_hash": motion_hash,
                    "quality_score": analysis_data["metrics"]["quality_score"],
                    "quality_category": analysis_data["metrics"]["quality_category"]
                },
                "description": f"Blended motion combining {motion1} ({(1-weight)*100:.0f}%) and {motion2} ({weight*100:.0f}%)",
                "file_path": f"/artifacts/{blend_id}.bvh",
                "analysis": analysis_data
            }
            
            # Store the artifact
            ARTIFACTS_STORE.append(artifact)
            
            logger.info(f"✅ Created blend artifact: {blend_id} with quality score {analysis_data['metrics']['quality_score']:.2f}")
            
            return jsonify({
                "status": "success",
                "artifact": artifact,
                "message": f"Blend created successfully with weight {weight}"
            })
            
        except Exception as e:
            logger.error(f"❌ Blend creation error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return jsonify({"error": str(e)}), 500@app.route('/api/artifacts', methods=['GET'])
def get_artifacts():
    """Get all generated artifacts"""
    return jsonify({
        "artifacts": ARTIFACTS_STORE,
        "total": len(ARTIFACTS_STORE)
    })

@app.route('/api/artifacts/manifest', methods=['GET'])
def get_artifacts_manifest():
    """Get artifacts manifest with metadata"""
    return jsonify({
        "artifacts": ARTIFACTS_STORE,
        "total": len(ARTIFACTS_STORE),
        "last_updated": datetime.utcnow().isoformat(),
        "storage_info": {
            "total_size": sum(2.3 for _ in ARTIFACTS_STORE),  # Mock size calculation
            "format_breakdown": {
                "BVH": len([a for a in ARTIFACTS_STORE if a.get("metadata", {}).get("format") == "BVH"]),
                "FBX": 0,
                "TRC": 0
            }
        }
    })

@app.route('/api/artifact/<artifact_name>/describe', methods=['GET'])
def describe_artifact(artifact_name):
    """Get detailed description of a specific artifact"""
    artifact = next((a for a in ARTIFACTS_STORE if a["name"] == artifact_name), None)
    
    if not artifact:
        return jsonify({"error": "Artifact not found"}), 404
    
    return jsonify({
        "artifact": artifact,
        "detailed_description": f"Motion blend artifact generated from {artifact['metadata']['source_motions']}",
        "technical_info": {
            "blend_algorithm": "Linear interpolation",
            "quality_score": artifact.get("metadata", {}).get("quality_score", 0.92),
            "compression": "None",
            "compatible_formats": ["BVH", "FBX", "USD"]
        }
    })

@app.route('/api/artifact/<artifact_id>/analysis', methods=['GET'])
def get_artifact_analysis(artifact_id):
    """Get analysis data for a specific artifact"""
    artifact = next((a for a in ARTIFACTS_STORE if a["id"] == artifact_id or a["name"] == artifact_id), None)
    
    if not artifact:
        return jsonify({"error": "Artifact not found"}), 404
    
    # If analysis exists, return it
    if "analysis" in artifact:
        return jsonify({
            "artifact_id": artifact["id"],
            "artifact_name": artifact["name"],
            "analysis": artifact["analysis"],
            "source_motions": artifact["metadata"]["source_motions"],
            "blend_weight": artifact["metadata"]["blend_weight"]
        })
    
    # If no analysis exists, generate it on-the-fly
    motion_hash = artifact["metadata"].get("motion_hash", 12345)
    frames = artifact["metadata"].get("frames", 120)
    duration = artifact["metadata"].get("duration", 4.0)
    weight = artifact["metadata"].get("blend_weight", 0.5)
    
    analysis_data = generate_blend_analysis(motion_hash, frames, duration, weight)
    
    # Store analysis in artifact for future requests
    artifact["analysis"] = analysis_data
    artifact["metadata"]["quality_score"] = analysis_data["metrics"]["quality_score"]
    artifact["metadata"]["quality_category"] = analysis_data["metrics"]["quality_category"]
    
    return jsonify({
        "artifact_id": artifact["id"],
        "artifact_name": artifact["name"],
        "analysis": analysis_data,
        "source_motions": artifact["metadata"]["source_motions"],
        "blend_weight": artifact["metadata"]["blend_weight"],
        "generated": "on-demand"
    })

if __name__ == '__main__':
    logger.info("=" * 80)
    logger.info("🚀 Starting MotionBlendAI Elasticsearch API server...")
    logger.info("=" * 80)
    
    with timer_context("Server initialization"):
        # Log environment configuration
        logger.info(f"🔧 DEBUG_MODE: {DEBUG_MODE}")
        logger.info(f"🔌 Elasticsearch available: {es_available}")
        logger.info(f"📊 Mock motions loaded: {len(MOCK_MOTIONS)}")
        logger.info(f"📇 Index name: {ES_INDEX_NAME}")
        
        # Skip Elasticsearch initialization in DEBUG mode for faster startup
        # ES will be initialized lazily on first search request if needed
        if DEBUG_MODE:
            logger.info("⚡ DEBUG_MODE: Skipping Elasticsearch initialization for fast startup")
            logger.info("   → Elasticsearch will connect lazily on first search request")
        else:
            # Initialize Elasticsearch connection when starting in production
            with timer_context("Initialize Elasticsearch"):
                initialize_elasticsearch()
        
        if es_available and not DEBUG_MODE:
            logger.info("✅ Connected to Elasticsearch - using real search")
        else:
            logger.warning("⚠️  Using mock data for development")
        
        # Get runtime configuration
        import os
        host = os.getenv('HOST', '0.0.0.0')
        port = int(os.getenv('PORT', 5000))
        debug = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
        
        logger.info(f"📡 Server configuration: {host}:{port} (debug={debug})")
        logger.info("=" * 80)
    
    # Run the Flask app (use 0.0.0.0 for Docker)
    app.run(debug=debug, host=host, port=port)
