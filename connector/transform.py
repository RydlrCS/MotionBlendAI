"""
Transform module: normalize and enrich motion records
"""
import hashlib
import time
from typing import Dict, Any
from logging_util import get_logger

logger = get_logger(__name__)

def generate_id(file_uri: str) -> str:
    """
    Generate deterministic ID from file URI
    
    Args:
        file_uri: GCS URI
        
    Returns:
        16-character hex ID
    """
    return hashlib.sha1(file_uri.encode()).hexdigest()[:16]


def normalize_seed(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform raw GCS record to seed_motions schema
    
    Args:
        record: Raw record from extract with file_uri, size, updated_at
        
    Returns:
        Normalized record matching seed_motions schema
    """
    rec_id = generate_id(record["file_uri"])
    
    normalized = {
        "id": rec_id,
        "file_uri": record["file_uri"],
        "skeleton_id": "mixamo24",  # Default skeleton
        "frames": None,  # Populated later from BVH parsing
        "fps": 30,  # Default FPS
        "joints_count": 24,  # Mixamo default
        "created_at": int(time.time()),
        "updated_at": int(time.time())
    }
    
    logger.info("normalized_seed", id=rec_id, uri=record["file_uri"])
    return normalized


def normalize_build(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform raw GCS record to build_motions schema
    
    Args:
        record: Raw record from extract
        
    Returns:
        Normalized record matching build_motions schema
    """
    rec_id = generate_id(record["file_uri"])
    
    normalized = {
        "id": rec_id,
        "file_uri": record["file_uri"],
        "skeleton_id": "mixamo24",
        "frames": None,
        "fps": 30,
        "joints_count": 24,
        "build_method": "ganimator",  # Default build method
        "created_at": int(time.time()),
        "updated_at": int(time.time())
    }
    
    logger.info("normalized_build", id=rec_id, uri=record["file_uri"])
    return normalized


def normalize_blend(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform raw GCS record to blend_snn schema
    
    Note: In production, left_motion_id and right_motion_id would come from
    metadata or filename parsing. Using placeholders for minimal setup.
    
    Args:
        record: Raw record from extract
        
    Returns:
        Normalized record matching blend_snn schema
    """
    rec_id = generate_id(record["file_uri"])
    
    # Extract potential motion IDs from filename (placeholder logic)
    # Real implementation would parse metadata or use naming convention
    normalized = {
        "id": rec_id,
        "left_motion_id": f"{rec_id}_left",  # Placeholder
        "right_motion_id": f"{rec_id}_right",  # Placeholder
        "blend_ratio": 0.5,  # Default 50/50 blend
        "transition_start_frame": 30,
        "transition_end_frame": 90,
        "method": "snn",  # Smooth Neural Network
        "created_at": int(time.time()),
        "updated_at": int(time.time())
    }
    
    logger.info("normalized_blend", id=rec_id, uri=record["file_uri"])
    return normalized


def normalize_record(record: Dict[str, Any], category: str) -> Dict[str, Any]:
    """
    Normalize record based on category
    
    Args:
        record: Raw record from extract
        category: One of 'seed', 'build', 'blend'
        
    Returns:
        Normalized record for appropriate table
    """
    if category == 'seed':
        return normalize_seed(record)
    elif category == 'build':
        return normalize_build(record)
    elif category == 'blend':
        return normalize_blend(record)
    else:
        raise ValueError(f"Unknown category: {category}")
