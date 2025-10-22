"""
Fivetran connector discovery: define schema catalog for BigQuery RAW tables
"""
from typing import Dict, List
from logging_util import get_logger

logger = get_logger(__name__)

def discover() -> Dict[str, List[str]]:
    """
    Return schema catalog for all tables
    
    Returns:
        Dictionary mapping table names to column lists
    """
    logger.info("discover_start", message="Generating schema catalog")
    
    catalog = {
        "seed_motions": [
            "id",
            "file_uri",
            "skeleton_id",
            "frames",
            "fps",
            "joints_count",
            "created_at",
            "updated_at"
        ],
        "build_motions": [
            "id",
            "file_uri",
            "skeleton_id",
            "frames",
            "fps",
            "joints_count",
            "build_method",
            "created_at",
            "updated_at"
        ],
        "blend_snn": [
            "id",
            "left_motion_id",
            "right_motion_id",
            "blend_ratio",
            "transition_start_frame",
            "transition_end_frame",
            "method",
            "created_at",
            "updated_at"
        ]
    }
    
    logger.info("discover_complete", tables=list(catalog.keys()), 
                total_columns=sum(len(cols) for cols in catalog.values()))
    
    return catalog
