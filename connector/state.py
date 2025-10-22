"""
State management: track sync progress and checkpoints
"""
import json
import os
from typing import Dict, Any
from datetime import datetime
from logging_util import get_logger

logger = get_logger(__name__)

DEFAULT_STATE_PATH = "connector_state.json"

def get_state(path: str = DEFAULT_STATE_PATH) -> Dict[str, Any]:
    """
    Load state from JSON file
    
    Args:
        path: Path to state file
        
    Returns:
        State dictionary or empty dict if file doesn't exist
    """
    if not os.path.exists(path):
        logger.info("state_not_found", path=path, message="Returning empty state")
        return {}
    
    try:
        with open(path, 'r') as f:
            state = json.load(f)
        logger.info("state_loaded", path=path, keys=list(state.keys()))
        return state
    except Exception as e:
        logger.error("state_load_failed", path=path, error=str(e))
        return {}


def set_state(state: Dict[str, Any], path: str = DEFAULT_STATE_PATH) -> None:
    """
    Save state to JSON file
    
    Args:
        state: State dictionary to save
        path: Path to state file
    """
    try:
        # Add metadata
        state["_last_updated"] = datetime.utcnow().isoformat()
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info("state_saved", path=path, keys=list(state.keys()))
    except Exception as e:
        logger.error("state_save_failed", path=path, error=str(e))
        raise


def update_cursor(
    category: str,
    cursor: Any,
    state: Dict[str, Any] = None,
    path: str = DEFAULT_STATE_PATH
) -> Dict[str, Any]:
    """
    Update cursor for a category
    
    Args:
        category: Motion category ('seed', 'build', 'blend')
        cursor: Cursor value (timestamp, file name, etc.)
        state: Existing state dict (loads from file if None)
        path: State file path
        
    Returns:
        Updated state dictionary
    """
    if state is None:
        state = get_state(path)
    
    if "cursors" not in state:
        state["cursors"] = {}
    
    state["cursors"][category] = {
        "value": cursor,
        "updated_at": datetime.utcnow().isoformat()
    }
    
    set_state(state, path)
    logger.info("cursor_updated", category=category, cursor=cursor)
    
    return state


def get_cursor(
    category: str,
    state: Dict[str, Any] = None,
    path: str = DEFAULT_STATE_PATH
) -> Any:
    """
    Get cursor for a category
    
    Args:
        category: Motion category
        state: Existing state dict (loads from file if None)
        path: State file path
        
    Returns:
        Cursor value or None if not found
    """
    if state is None:
        state = get_state(path)
    
    cursor_data = state.get("cursors", {}).get(category)
    if cursor_data:
        logger.info("cursor_found", category=category, 
                   cursor=cursor_data.get("value"))
        return cursor_data.get("value")
    
    logger.info("cursor_not_found", category=category)
    return None


def record_sync(
    category: str,
    records_processed: int,
    state: Dict[str, Any] = None,
    path: str = DEFAULT_STATE_PATH
) -> Dict[str, Any]:
    """
    Record sync statistics
    
    Args:
        category: Motion category
        records_processed: Number of records processed
        state: Existing state dict
        path: State file path
        
    Returns:
        Updated state dictionary
    """
    if state is None:
        state = get_state(path)
    
    if "sync_history" not in state:
        state["sync_history"] = {}
    
    if category not in state["sync_history"]:
        state["sync_history"][category] = {
            "total_records": 0,
            "sync_count": 0,
            "last_sync": None
        }
    
    history = state["sync_history"][category]
    history["total_records"] += records_processed
    history["sync_count"] += 1
    history["last_sync"] = datetime.utcnow().isoformat()
    
    set_state(state, path)
    logger.info("sync_recorded", category=category, 
               records=records_processed, 
               total=history["total_records"])
    
    return state
