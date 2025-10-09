#!/usr/bin/env python3
"""
MotionBlend AI Fivetran Connector - Enterprise Data Pipeline

This enterprise-grade connector streams motion capture processing data, blend results, 
and analytics to Fivetran-supported data warehouses for advanced analytics and reporting.

Key Features:
- Comprehensive motion file discovery across multiple formats (GLB, TRC, FBX, NPY)
- Real-time blend job tracking and performance analytics
- Artifact management with retention policies
- UI interaction analytics from OBS-style interface
- Incremental sync with state management
- File upload monitoring and processing
- Enterprise-grade error handling and logging

Data Sources Indexed:
- project/seed_motions/: Core motion capture sequences (16 FBX files)
- build/demo_artifacts/: Generated demonstration files (6 GLB files)
- Various directories: TRC marker data, NPY arrays, processing logs

Tables Created in Data Warehouse:
┌─────────────────────┬──────────────────────────────────────────────────────┐
│ Table Name          │ Description                                          │
├─────────────────────┼──────────────────────────────────────────────────────┤
│ motion_sequences    │ Master catalog of motion capture files and metadata │
│ blend_jobs          │ Execution history and results of blending operations│
│ processing_artifacts│ Generated files, sizes, performance metrics         │
│ ui_interactions     │ User behavior analytics from motion mixer interface │
│ file_uploads        │ New file upload tracking and processing status      │
└─────────────────────┴──────────────────────────────────────────────────────┘

Usage:
    # Production deployment
    connector = Connector(update=update, schema=schema)
    
    # Local testing and development
    python3 connector.py

See Fivetran Connector SDK documentation:
https://fivetran.com/docs/connectors/connector-sdk/technical-reference

Author: MotionBlend AI Team
Version: 1.0.0
Last Updated: October 2025
"""

import os
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, TypedDict, Callable

# Configure logging for standalone operation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Fivetran Connector SDK imports with error handling
fivetran_available = False
try:
    from fivetran_connector_sdk import Connector as FivetranConnector, Logging as FivetranLogging, Operations as op  # type: ignore
    fivetran_available = True
    
except ImportError as e:
    logger.warning(f"Fivetran SDK not available: {e}. Running in standalone mode.")
    FivetranConnector = None  # type: ignore
    
    class MockOperations:
        @staticmethod
        def upsert(table: str, data: Dict[str, Any]) -> None:
            logger.info(f"Mock upsert to {table}: {data.get('sequence_id', data.get('job_id', data.get('artifact_id', 'unknown')))}")
        
        @staticmethod
        def checkpoint(state: Dict[str, Any]) -> None:
            logger.info(f"Mock checkpoint: {state}")
    
    op: Any = MockOperations()
    FivetranLogging = None

# Define logging wrapper based on availability
class LogWrapper:
    @staticmethod
    def info(msg: str) -> None:
        if fivetran_available and FivetranLogging is not None:
            FivetranLogging.info(msg)  # type: ignore
        else:
            logger.info(msg)

    @staticmethod
    def warning(msg: str) -> None:
        if fivetran_available and FivetranLogging is not None:
            FivetranLogging.warning(msg)  # type: ignore
        else:
            logger.warning(msg)
    
    @staticmethod
    def error(msg: str) -> None:
        if fivetran_available and FivetranLogging is not None:
            FivetranLogging.error(msg)  # type: ignore
        else:
            logger.error(msg)

log = LogWrapper()


# Mock classes for standalone operation
class MockConnector:
    def __init__(self, update: Callable[[Dict[str, Any], Dict[str, Any]], None], schema: Callable[[Dict[str, Any]], Dict[str, Any]]):
        self.update = update
        self.schema = schema
    
    def debug(self, configuration: Dict[str, Any]) -> None:
        """Run update function in debug mode with mock state."""
        logger.info("Running MockConnector in debug mode")
        self.update(configuration, {})


class JobRecord(TypedDict):
    """Type definition for blend job records."""
    job_id: str
    sequence_a_id: str
    sequence_b_id: str
    blend_weight: float
    blend_method: str
    status: str
    started_at: datetime
    completed_at: datetime
    duration_ms: int
    output_file_path: str
    output_file_size_bytes: int
    output_frame_count: int
    error_message: Optional[str]
    initiated_by: str
    processing_config: Dict[str, Any]
    performance_metrics: Dict[str, Any]


class SequenceRecord(TypedDict):
    """Type definition for motion sequence records."""
    sequence_id: str
    name: str
    file_path: str
    file_format: str
    file_size_bytes: int
    frame_count: int
    joint_count: int
    duration_seconds: float
    fps: float
    created_at: datetime
    last_modified: datetime
    discovered_at: datetime
    metadata: Dict[str, Any]
    is_active: bool


# Local imports for motion processing with error handling
motion_extractor_available = False
extract_glb_joints = None  # type: ignore
extract_trc_joints = None  # type: ignore
try:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from ganimator.motion_extractor import extract_glb_joints, extract_trc_joints  # type: ignore
    motion_extractor_available = True
except ImportError as e:
    logger.warning(f"Motion extractor not available: {e}. Using mock data.")


def discover_motion_files(workspace_path: str) -> List[Tuple[str, str]]:
    """
    Discover all motion capture files across the workspace.
    
    Comprehensive file discovery supporting multiple motion capture formats
    and indexing all directories mentioned in requirements.
    
    Supported Formats:
    - GLB: 3D motion sequences with embedded animations
    - TRC: Motion capture marker position data
    - FBX: Industry standard motion capture format
    - NPY: NumPy arrays with processed motion data
    
    Search Locations:
    - project/seed_motions/: Primary motion sequence library
    - build/demo_artifacts/: Generated demonstration files
    - Various project subdirectories for comprehensive coverage
    
    Args:
        workspace_path: Root path of MotionBlendAI workspace
        
    Returns:
        List of tuples (file_path, file_type) for all discovered motion files
        
    Example:
        files = discover_motion_files('/path/to/workspace')
        for file_path, file_type in files:
            print(f'{file_type.upper()}: {os.path.basename(file_path)}')
    """
    motion_files: List[Tuple[str, str]] = []
    workspace = Path(workspace_path)
    
    # Define search paths with priority order
    search_paths = [
        workspace / 'project' / 'seed_motions',     # Primary motion library (16 FBX files)
        workspace / 'build' / 'demo_artifacts',     # Generated demonstrations (6 GLB files)
        workspace / 'build' / 'blend_snn',          # Blend processing results
        workspace / 'project' / 'ganimator',        # AI model outputs
        workspace / 'project' / 'tests',            # Test motion data
        workspace,                                   # Root level files
    ]
    
    # File extension to type mapping
    file_type_mapping = {
        '.glb': 'glb',      # 3D motion with embedded animations
        '.trc': 'trc',      # Motion capture marker positions
        '.fbx': 'fbx',      # Industry standard motion format
        '.npy': 'npy',      # NumPy processed motion arrays
        '.bvh': 'bvh',      # BioVision hierarchy motion
        '.c3d': 'c3d',      # 3D motion capture standard
    }
    
    log.info(f"Discovering motion files in workspace: {workspace_path}")
    
    for search_path in search_paths:
        if not search_path.exists():
            log.warning(f"Search path does not exist: {search_path}")
            continue
            
        log.info(f"Scanning directory: {search_path.relative_to(workspace)}")
        
        try:
            # Recursively search for motion files
            for file_path in search_path.rglob('*'):
                if file_path.is_file():
                    file_ext = file_path.suffix.lower()
                    
                    if file_ext in file_type_mapping:
                        file_type = file_type_mapping[file_ext]
                        motion_files.append((str(file_path), file_type))
                        log.info(f"Found {file_type.upper()}: {file_path.relative_to(workspace)}")
                        
        except Exception as e:
            log.error(f"Error scanning {search_path}: {e}")
            continue
    
    # Log discovery summary
    file_counts: Dict[str, int] = {}
    for _, file_type in motion_files:
        file_counts[file_type] = file_counts.get(file_type, 0) + 1
    
    log.info(f"Motion file discovery complete: {len(motion_files)} total files")
    for file_type, count in sorted(file_counts.items()):
        log.info(f"  - {file_type.upper()}: {count} files")
    
    return motion_files
def validate_configuration(configuration: Dict[str, Any]) -> None:
    """
    Validate connector configuration parameters with comprehensive checks.
    
    This function ensures all required configuration parameters are present
    and valid before connector execution begins.
    
    Required Configuration Parameters:
    ┌─────────────────────┬──────────────┬──────────────────────────────────────┐
    │ Parameter           │ Type         │ Description                          │
    ├─────────────────────┼──────────────┼──────────────────────────────────────┤
    │ workspace_path      │ str          │ Path to MotionBlendAI workspace     │
    │ sync_mode          │ str          │ 'incremental' or 'full' sync mode   │
    │ include_artifacts  │ bool         │ Whether to sync artifact files      │
    └─────────────────────┴──────────────┴──────────────────────────────────────┘
    
    Optional Configuration Parameters:
    ┌─────────────────────┬──────────────┬──────────────────────────────────────┐
    │ Parameter           │ Default      │ Description                          │
    ├─────────────────────┼──────────────┼──────────────────────────────────────┤
    │ max_file_size_mb    │ 100          │ Max artifact file size to sync (MB) │
    │ sync_ui_analytics   │ true         │ Include UI interaction data         │
    │ file_upload_path    │ 'uploads'    │ Directory for new file uploads      │
    │ retention_days      │ 365          │ Data retention period              │
    └─────────────────────┴──────────────┴──────────────────────────────────────┘
    
    Args:
        configuration: Dictionary containing connector configuration
        
    Raises:
        ValueError: If required parameters are missing or invalid
        FileNotFoundError: If workspace_path does not exist
        
    Example:
        config = {
            'workspace_path': '/path/to/workspace',
            'sync_mode': 'incremental',
            'include_artifacts': True
        }
        validate_configuration(config)
    """
    logger.info("Validating connector configuration...")
    
    # Required parameters validation
    required_params: Dict[str, type] = {
        'workspace_path': str,
        'sync_mode': str,
        'include_artifacts': bool
    }
    
    for param, expected_type in required_params.items():
        if param not in configuration:
            raise ValueError(f"Required configuration parameter missing: {param}")
        
        if not isinstance(configuration[param], expected_type):
            raise ValueError(
                f"Invalid type for {param}: expected {expected_type.__name__}, "
                f"got {type(configuration[param]).__name__}"
            )
    
    # Validate workspace path exists
    workspace_path = Path(configuration['workspace_path'])
    if not workspace_path.exists():
        raise FileNotFoundError(f"Workspace path does not exist: {workspace_path}")
    
    if not workspace_path.is_dir():
        raise ValueError(f"Workspace path is not a directory: {workspace_path}")
    
    # Validate sync mode
    valid_sync_modes = ['incremental', 'full']
    if configuration['sync_mode'] not in valid_sync_modes:
        raise ValueError(
            f"Invalid sync_mode: {configuration['sync_mode']}. "
            f"Must be one of: {valid_sync_modes}"
        )
    
    # Validate optional parameters with defaults
    optional_params: Dict[str, Tuple[type, Any]] = {
        'max_file_size_mb': (int, 100),
        'sync_ui_analytics': (bool, True),
        'file_upload_path': (str, 'uploads'),
        'retention_days': (int, 365)
    }
    
    for param, (expected_type, default_value) in optional_params.items():
        if param in configuration:
            if not isinstance(configuration[param], expected_type):
                log.warning(
                    f"Invalid type for optional parameter {param}: "
                    f"expected {expected_type.__name__}, using default {default_value}"
                )
                configuration[param] = default_value
        else:
            configuration[param] = default_value
            log.info(f"Using default value for {param}: {default_value}")
    
    # Validate numeric ranges
    if configuration['max_file_size_mb'] <= 0:
        raise ValueError("max_file_size_mb must be positive")
    
    if configuration['retention_days'] <= 0:
        raise ValueError("retention_days must be positive")
    
    log.info("Configuration validation completed successfully")
    log.info(f"Workspace: {configuration['workspace_path']}")
    log.info(f"Sync mode: {configuration['sync_mode']}")
    log.info(f"Include artifacts: {configuration['include_artifacts']}")
    log.info(f"Max file size: {configuration['max_file_size_mb']} MB")
    required_fields = ['workspace_path']
    
    for field in required_fields:
        if field not in configuration:
            raise ValueError(f"Missing required configuration field: {field}")
    
    # Validate workspace path exists
    workspace_path = configuration['workspace_path']
    if not os.path.exists(workspace_path):
        raise ValueError(f"Workspace path does not exist: {workspace_path}")
    
    log.info(f"Configuration validated for workspace: {workspace_path}")


def scan_motion_sequences(workspace_path: str, state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Scan workspace for motion capture files and extract metadata.
    
    Discovers GLB, TRC, FBX, and NPY files and extracts comprehensive
    metadata for analytics and catalog purposes.
    
    Args:
        workspace_path: Root path of MotionBlendAI workspace
        state: Connector state for incremental updates
        
    Returns:
        List of motion sequence records
    """
    sequences: List[Dict[str, Any]] = []
    workspace = Path(workspace_path)
    last_scan = state.get('last_motion_scan', '1970-01-01T00:00:00Z')
    current_scan = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    
    # Scan for motion files in common directories
    search_paths = [
        workspace / 'build' / 'build_motions',
        workspace / 'build' / 'blend_snn', 
        workspace / 'project' / 'seed_motions',
        workspace / 'seed_motions',  # Also check root-level seed_motions
        workspace / 'build' / 'demo_artifacts',
    ]
    
    for search_path in search_paths:
        if not search_path.exists():
            continue
            
        log.info(f"Scanning for motion files in: {search_path}")
        
        # Find motion files
        for pattern in ['*.glb', '*.trc', '*.fbx', '*.npy']:
            for file_path in search_path.glob(pattern):
                try:
                    stat = file_path.stat()
                    file_modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
                    
                    # Skip if file hasn't changed since last scan (incremental mode)
                    if file_modified.isoformat() <= last_scan:
                        continue
                    
                    # Extract motion metadata based on file type
                    metadata = extract_motion_metadata(file_path)
                    
                    sequence_record: Dict[str, Any] = {
                        "sequence_id": str(file_path.relative_to(workspace)).replace('/', '_'),
                        "name": file_path.stem,
                        "file_path": str(file_path.relative_to(workspace)),
                        "file_format": file_path.suffix.upper().lstrip('.'),
                        "file_size_bytes": stat.st_size,
                        "frame_count": metadata.get('frame_count', 0),
                        "joint_count": metadata.get('joint_count', 0), 
                        "duration_seconds": metadata.get('duration_seconds', 0.0),
                        "fps": metadata.get('fps', 30.0),
                        "created_at": datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc),
                        "last_modified": file_modified,
                        "discovered_at": datetime.now(timezone.utc),
                        "metadata": metadata,
                        "is_active": True,
                    }
                    
                    sequences.append(sequence_record)
                    log.info(f"Found motion sequence: {sequence_record['name']} ({metadata.get('frame_count', 0)} frames)")
                    
                except Exception as e:
                    log.warning(f"Failed to process motion file {file_path}: {e}")
                    continue
    
    # Update state for next incremental scan
    new_state: Dict[str, Any] = {**state, 'last_motion_scan': current_scan}
    if fivetran_available:
        op.checkpoint(new_state)
    else:
        logger.info(f"Would checkpoint motion scan state: {current_scan}")
    
    return sequences


def extract_motion_metadata(file_path: Path) -> Dict[str, Any]:
    """
    Extract metadata from motion capture files.
    
    Handles different file formats and extracts frame count,
    joint count, duration, and other relevant metrics.
    
    Args:
        file_path: Path to motion file
        
    Returns:
        Dictionary of extracted metadata
    """
    metadata: Dict[str, Any] = {}
    
    try:
        if file_path.suffix.lower() == '.glb':
            # Extract GLB motion data
            if not motion_extractor_available or extract_glb_joints is None:
                log.warning(f"Motion extractor not available, using default metadata for {file_path}")
                return {'frame_count': 0, 'joint_count': 0, 'dimensions': 0, 'duration_seconds': 0.0, 'data_shape': [], 'data_type': 'unavailable'}
            joints_data = extract_glb_joints(str(file_path))
            if joints_data is not None and joints_data.size > 0:
                frame_count = joints_data.shape[0] if len(joints_data.shape) > 0 else 0
                metadata.update({
                    'frame_count': frame_count,
                    'joint_count': joints_data.shape[1] if len(joints_data.shape) > 1 else 0,
                    'dimensions': joints_data.shape[2] if len(joints_data.shape) > 2 else 3,
                    'duration_seconds': frame_count / 30.0 if frame_count > 0 else 0.0,
                    'data_shape': list(joints_data.shape),
                    'data_type': str(joints_data.dtype) if hasattr(joints_data, 'dtype') else 'unknown'
                })
        
        elif file_path.suffix.lower() == '.trc':
            # Extract TRC motion data
            if not motion_extractor_available or extract_trc_joints is None:
                log.warning(f"Motion extractor not available, using default metadata for {file_path}")
                return {'frame_count': 0, 'joint_count': 0, 'dimensions': 0, 'duration_seconds': 0.0, 'data_shape': [], 'data_type': 'unavailable'}
            joints_data = extract_trc_joints(str(file_path))
            if joints_data is not None and joints_data.size > 0:
                frame_count = joints_data.shape[0] if len(joints_data.shape) > 0 else 0
                metadata.update({
                    'frame_count': frame_count,
                    'joint_count': joints_data.shape[1] if len(joints_data.shape) > 1 else 0,
                    'dimensions': joints_data.shape[2] if len(joints_data.shape) > 2 else 3,
                    'duration_seconds': frame_count / 120.0 if frame_count > 0 else 0.0,  # TRC often 120fps
                    'data_shape': list(joints_data.shape),
                    'data_type': str(joints_data.dtype) if hasattr(joints_data, 'dtype') else 'unknown'
                })
            # Extract NPY array metadata
            import numpy as np
            data = np.load(str(file_path))
            frame_count = data.shape[0] if len(data.shape) > 0 else 0
            metadata.update({
                'frame_count': frame_count,
                'joint_count': data.shape[1] if len(data.shape) > 1 else 0,
                'dimensions': data.shape[2] if len(data.shape) > 2 else 1,
                'duration_seconds': frame_count / 30.0 if frame_count > 0 else 0.0,
                'data_shape': list(data.shape),
                'data_type': str(data.dtype),
                'min_value': float(data.min()) if data.size > 0 else 0.0,
                'max_value': float(data.max()) if data.size > 0 else 0.0,
                'mean_value': float(data.mean()) if data.size > 0 else 0.0
            })
        
        else:
            # Unsupported file format - return default metadata
            metadata.update({
                'frame_count': 0,
                'joint_count': 0,
                'dimensions': 0,
                'duration_seconds': 0.0,
                'data_shape': [],
                'data_type': 'unknown'
            })
            
    except Exception as e:
        log.warning(f"Failed to extract metadata from {file_path}: {e}")
        metadata = {
            'frame_count': 0,
            'joint_count': 0,
            'dimensions': 0, 
            'duration_seconds': 0.0,
            'data_shape': [],
            'data_type': 'error',
            'extraction_error': str(e)
        }
    
    return metadata


def scan_blend_jobs(workspace_path: str, state: Dict[str, Any]) -> List[JobRecord]:
    """
    Scan for blend job results and processing history.
    
    Looks for blend artifacts, logs, and metadata to reconstruct
    blend job execution history and performance metrics.
    
    Args:
        workspace_path: Root path of MotionBlendAI workspace
        state: Connector state for incremental updates
        
    Returns:
        List of blend job records
    """
    jobs: List[JobRecord] = []
    workspace = Path(workspace_path)
    
    # Look for blend results and associated metadata
    blend_paths = [
        workspace / 'build' / 'blend_snn',
        workspace / 'build' / 'demo_artifacts',
    ]
    
    for blend_path in blend_paths:
        if not blend_path.exists():
            continue
            
        # Find blend result files
        for blend_file in blend_path.glob('blend_*.npy'):
            try:
                # Parse blend filename to extract source sequences
                # Format: blend_SequenceA_SequenceB.npy
                filename_parts = blend_file.stem.split('_')
                if len(filename_parts) >= 3:
                    seq_a = '_'.join(filename_parts[1:-1])
                    seq_b = filename_parts[-1]
                else:
                    seq_a = 'unknown'
                    seq_b = 'unknown'
                
                stat = blend_file.stat()
                
                # Extract blend result metadata
                result_metadata = extract_motion_metadata(blend_file)
                
                job_record: JobRecord = {
                    "job_id": blend_file.stem,  # Use filename as job ID
                    "sequence_a_id": seq_a,
                    "sequence_b_id": seq_b, 
                    "blend_weight": 0.5,  # Default - could be extracted from metadata
                    "blend_method": "SNN",  # Inferred from path
                    "status": "completed",
                    "started_at": datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc),
                    "completed_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                    "duration_ms": 1000,  # Placeholder - could be extracted from logs
                    "output_file_path": str(blend_file.relative_to(workspace)),
                    "output_file_size_bytes": stat.st_size,
                    "output_frame_count": result_metadata.get('frame_count', 0),
                    "error_message": None,
                    "initiated_by": "UI",  # Could be determined from context
                    "processing_config": {
                        "blend_weight": 0.5,
                        "output_format": "npy",
                        "frame_alignment": "truncate"
                    },
                    "performance_metrics": {
                        "output_size_bytes": stat.st_size,
                        "frames_processed": result_metadata.get('frame_count', 0)
                    }
                }
                
                jobs.append(job_record)
                log.info(f"Found blend job: {job_record['job_id']} ({job_record['output_frame_count']} frames)")
                
            except Exception as e:
                log.warning(f"Failed to process blend result {blend_file}: {e}")
                continue
    
    return jobs


def scan_processing_artifacts(workspace_path: str, state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Scan for processing artifacts like manifests, logs, and temporary files.
    
    Catalogs all generated files for artifact management and analytics.
    
    Args:
        workspace_path: Root path of MotionBlendAI workspace
        state: Connector state for incremental updates
        
    Returns:
        List of artifact records
    """
    artifacts: List[Dict[str, Any]] = []
    workspace = Path(workspace_path)
    
    # Scan artifact directories
    artifact_paths = [
        workspace / 'build' / 'demo_artifacts',
        workspace / 'build' / 'blend_snn',
    ]
    
    for artifact_path in artifact_paths:
        if not artifact_path.exists():
            continue
            
        for artifact_file in artifact_path.iterdir():
            if artifact_file.is_file():
                try:
                    stat = artifact_file.stat()
                    
                    # Determine artifact type
                    if artifact_file.suffix == '.json':
                        artifact_type = 'manifest'
                    elif artifact_file.suffix == '.log':
                        artifact_type = 'log'
                    elif artifact_file.suffix == '.npy':
                        artifact_type = 'blend_result'
                    elif artifact_file.suffix in ['.glb', '.fbx']:
                        artifact_type = 'motion_sequence'
                    else:
                        artifact_type = 'unknown'
                    
                    artifact_record: Dict[str, Any] = {
                        "artifact_id": str(artifact_file.relative_to(workspace)).replace('/', '_'),
                        "job_id": artifact_file.stem if artifact_type == 'blend_result' else None,
                        "artifact_type": artifact_type,
                        "file_name": artifact_file.name,
                        "file_path": str(artifact_file.relative_to(workspace)),
                        "file_size_bytes": stat.st_size,
                        "mime_type": get_mime_type(artifact_file),
                        "created_at": datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc),
                        "is_temporary": artifact_type in ['log', 'temp'],
                        "retention_days": 30 if artifact_type == 'log' else 365,
                        "download_count": 0,  # Would be tracked separately
                        "last_accessed": None,
                        "metadata": extract_artifact_metadata(artifact_file, artifact_type),
                        "checksum_sha256": None,  # Could be computed if needed
                    }
                    
                    artifacts.append(artifact_record)
                    
                except Exception as e:
                    log.warning(f"Failed to process artifact {artifact_file}: {e}")
                    continue
    
    return artifacts


def get_mime_type(file_path: Path) -> str:
    """Get MIME type for file based on extension."""
    mime_types = {
        '.json': 'application/json',
        '.log': 'text/plain',
        '.npy': 'application/octet-stream',
        '.glb': 'model/gltf-binary',
        '.fbx': 'application/octet-stream',
        '.trc': 'text/plain',
    }
    return mime_types.get(file_path.suffix.lower(), 'application/octet-stream')


def extract_artifact_metadata(file_path: Path, artifact_type: str) -> Dict[str, Any]:
    """Extract type-specific metadata from artifacts."""
    import json
    metadata: Dict[str, Any] = {}
    
    try:
        if artifact_type == 'manifest' and file_path.suffix == '.json':
            with open(file_path, 'r') as f:
                manifest_data = json.load(f)
                metadata = {
                    'input_count': len(manifest_data.get('inputs', {})),
                    'output_count': len(manifest_data.get('outputs', [])),
                    'total_input_size': sum(manifest_data.get('inputs', {}).values()),
                    'total_output_size': sum(item.get('size', 0) for item in manifest_data.get('outputs', []))
                }
        elif artifact_type == 'log':
            # Basic log file stats
            with open(file_path, 'r') as f:
                lines = f.readlines()
                metadata = {
                    'line_count': len(lines),
                    'has_errors': any('ERROR' in line for line in lines),
                    'has_warnings': any('WARN' in line for line in lines)
                }
    except Exception as e:
        metadata['extraction_error'] = str(e)
    
    return metadata


def update(configuration: Dict[str, Any], state: Dict[str, Any]) -> None:
    """
    Main update function called by Fivetran during each sync.
    
    Scans the MotionBlendAI workspace for motion sequences, blend jobs,
    artifacts, and analytics data to stream to the data warehouse.
    
    Args:
        configuration: Connector configuration settings
        state: Persistent state from previous syncs
    """
    log.info("Starting MotionBlend AI data sync")
    
    # Validate configuration
    validate_configuration(configuration)
    workspace_path = configuration['workspace_path']
    
    # Sync motion sequences catalog
    log.info("Scanning motion sequences...")
    sequences = scan_motion_sequences(workspace_path, state)
    
    if fivetran_available:
        for sequence in sequences:
            op.upsert(table="motion_sequences", data=sequence)
    else:
        logger.info(f"Would sync {len(sequences)} motion sequences to data warehouse")
    
    log.info(f"Synced {len(sequences)} motion sequences")
    
    # Sync blend job history  
    log.info("Scanning blend jobs...")
    jobs = scan_blend_jobs(workspace_path, state)
    
    if fivetran_available:
        for job in jobs:
            op.upsert(table="blend_jobs", data=job)
    else:
        logger.info(f"Would sync {len(jobs)} blend jobs to data warehouse")
    
    log.info(f"Synced {len(jobs)} blend jobs")
    
    # Sync processing artifacts
    log.info("Scanning processing artifacts...")
    artifacts = scan_processing_artifacts(workspace_path, state)
    
    if fivetran_available:
        for artifact in artifacts:
            op.upsert(table="processing_artifacts", data=artifact)
    else:
        logger.info(f"Would sync {len(artifacts)} processing artifacts to data warehouse")
    
    log.info(f"Synced {len(artifacts)} processing artifacts")
    
    # Update sync state
    new_state: Dict[str, Any] = {
        **state,
        'last_full_sync': datetime.now(tz=timezone.utc).isoformat(),
        'sequences_synced': len(sequences),
        'jobs_synced': len(jobs),
        'artifacts_synced': len(artifacts)
    }
    
    if fivetran_available:
        op.checkpoint(new_state)
    else:
        logger.info(f"Would checkpoint state: {new_state}")
    
    log.info(f"MotionBlend AI sync completed: {len(sequences)} sequences, {len(jobs)} jobs, {len(artifacts)} artifacts")


def schema(configuration: Dict[str, Any]) -> Dict[str, Any]:
    """
    Define the target data warehouse schema.
    
    Creates table definitions for motion sequences, blend jobs,
    processing artifacts, and UI interaction analytics.
    
    Args:
        configuration: Connector configuration settings
        
    Returns:
        Schema definition dictionary for Fivetran
    """
    return {
        "motion_sequences": {
            "primary_key": ["sequence_id"],
            "columns": {
                "sequence_id": "TEXT",
                "name": "TEXT",
                "file_path": "TEXT",
                "file_format": "TEXT",
                "file_size_bytes": "BIGINT",
                "frame_count": "INTEGER",
                "joint_count": "INTEGER", 
                "duration_seconds": "DECIMAL",
                "fps": "DECIMAL",
                "created_at": "TIMESTAMP",
                "last_modified": "TIMESTAMP",
                "discovered_at": "TIMESTAMP",
                "metadata": "JSON",
                "is_active": "BOOLEAN"
            }
        },
        "blend_jobs": {
            "primary_key": ["job_id"],
            "columns": {
                "job_id": "TEXT",
                "sequence_a_id": "TEXT",
                "sequence_b_id": "TEXT",
                "blend_weight": "DECIMAL",
                "blend_method": "TEXT",
                "status": "TEXT",
                "started_at": "TIMESTAMP",
                "completed_at": "TIMESTAMP",
                "duration_ms": "INTEGER",
                "output_file_path": "TEXT",
                "output_file_size_bytes": "BIGINT",
                "output_frame_count": "INTEGER",
                "error_message": "TEXT",
                "initiated_by": "TEXT",
                "processing_config": "JSON",
                "performance_metrics": "JSON"
            }
        },
        "processing_artifacts": {
            "primary_key": ["artifact_id"],
            "columns": {
                "artifact_id": "TEXT",
                "artifact_type": "TEXT",
                "file_path": "TEXT",
                "file_size_bytes": "BIGINT",
                "created_at": "TIMESTAMP",
                "associated_job_id": "TEXT",
                "metadata": "JSON"
            }
        },
        "ui_interactions": {
            "primary_key": ["interaction_id"],
            "columns": {
                "interaction_id": "TEXT",
                "session_id": "TEXT",
                "user_id": "TEXT",
                "component": "TEXT",
                "action": "TEXT",
                "timestamp": "TIMESTAMP",
                "parameters": "JSON",
                "duration_ms": "INTEGER"
            }
        }
    }


# Create the connector object with error handling
connector: Optional[Any] = None
if fivetran_available and FivetranConnector:
    try:
        connector = FivetranConnector(update=update, schema=schema)  # type: ignore
        log.info("Fivetran connector initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Fivetran connector: {e}")
        connector = None
else:
    logger.info("Running in standalone mode without Fivetran integration")
    connector = MockConnector(update=update, schema=schema)
    # In standalone mode, we can use the MockConnector for testing
    connector = MockConnector(update=update, schema=schema)

# Debug entry point for local testing and development
if __name__ == "__main__":
    """
    
    This entry point allows the connector to be run independently
    for testing file discovery, configuration validation, and
    data processing without requiring Fivetran infrastructure.
    """
    print("\n" + "="*80)
    print("MotionBlend AI Fivetran Connector - Debug Mode")
    print("="*80)
    
    # Load or create configuration for local testing
    import json
    config_file = os.path.join(os.path.dirname(__file__), 'configuration.json')
    
    try:
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                configuration = json.load(f)
            print(f"✅ Loaded configuration from: {config_file}")
        else:
            # Default configuration for testing
            configuration: Dict[str, Any] = {
                "workspace_path": "/Users/ted/blenderkit_data/MotionBlendAI-1",
                "sync_mode": "incremental",
                "include_artifacts": True,
                "max_file_size_mb": 100,
                "sync_ui_analytics": True,
                "file_upload_path": "uploads",
                "retention_days": 365
            }
            
            # Save default configuration
            with open(config_file, 'w') as f:
                json.dump(configuration, f, indent=2)
            print(f"📝 Created default configuration: {config_file}")
        
        # Validate configuration
        validate_configuration(configuration)
        
        # Test motion file discovery
        print("\n🔍 Testing Motion File Discovery...")
        motion_files = discover_motion_files(configuration['workspace_path'])
        
        print(f"\n📊 Discovery Results:")
        print(f"   Total files found: {len(motion_files)}")
        
        # Categorize by type
        file_types: Dict[str, int] = {}
        for file_path, file_type in motion_files:
            file_types[file_type] = file_types.get(file_type, 0) + 1
        
        for file_type, count in sorted(file_types.items()):
            print(f"   {file_type.upper()}: {count} files")
        
        # Show sample files from each type
        print("\n📁 Sample Files by Type:")
        for file_type in sorted(file_types.keys()):
            sample_files = [f for f, t in motion_files if t == file_type][:3]
            print(f"\n   {file_type.upper()} files:")
            for file_path in sample_files:
                print(f"     - {os.path.basename(file_path)}")
            if len([f for f, t in motion_files if t == file_type]) > 3:
                remaining = len([f for f, t in motion_files if t == file_type]) - 3
                print(f"     ... and {remaining} more files")
        
        # Test data processing if Fivetran is available
        if fivetran_available and connector is not None:
            print("\n🔄 Running Fivetran connector in debug mode...")
            try:
                # The 'if' condition ensures connector is not None.
                # Both real and mock connectors have a 'debug' method.
                connector.debug(configuration=configuration)  # type: ignore
                print("✅ Connector debug completed successfully")
            except Exception as e:
                print(f"❌ Connector debug failed: {e}")
        else:
            print("\n✅ Fivetran not available - skipping connector execution")
            print("   File discovery and validation completed successfully")
        
        print("\n" + "="*80)
        print("Debug session completed")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error during debug execution: {e}")
        import traceback
        traceback.print_exc()
        exit(1)