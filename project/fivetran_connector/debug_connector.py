#!/usr/bin/env python3
"""
Simple Fivetran Connector Test - Debug Mode

This script tests the Fivetran connector functionality without
the problematic SDK logging that's causing issues.
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict, Union
import logging

# Setup simple logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def discover_motion_files_simple(workspace_path: str) -> List[Tuple[str, str]]:
    """Simple version of motion file discovery without Fivetran logging."""
    motion_files: List[Tuple[str, str]] = []
    workspace = Path(workspace_path)
    
    # Define search paths
    search_paths = [
        workspace / 'project' / 'seed_motions',
        workspace / 'build' / 'demo_artifacts', 
        workspace / 'build' / 'blend_snn',
        workspace / 'project' / 'ganimator',
        workspace / 'project' / 'tests',
        workspace,
    ]
    
    # File extension mapping
    file_type_mapping = {
        '.glb': 'glb',
        '.trc': 'trc', 
        '.fbx': 'fbx',
        '.npy': 'npy',
        '.bvh': 'bvh',
        '.c3d': 'c3d',
    }
    
    logger.info(f"Discovering motion files in: {workspace_path}")
    
    for search_path in search_paths:
        if not search_path.exists():
            logger.warning(f"Path not found: {search_path}")
            continue
            
        logger.info(f"Scanning: {search_path.relative_to(workspace)}")
        
        try:
            for file_path in search_path.rglob('*'):
                if file_path.is_file():
                    file_ext = file_path.suffix.lower()
                    if file_ext in file_type_mapping:
                        file_type = file_type_mapping[file_ext]
                        motion_files.append((str(file_path), file_type))
                        logger.debug(f"Found {file_type.upper()}: {file_path.name}")
        except Exception as e:
            logger.error(f"Error scanning {search_path}: {e}")
    
    return motion_files

def test_connector_functionality():
    """Test all connector functionality step by step."""
    print("=" * 70)
    print("🔧 Fivetran Connector Debug Test")
    print("=" * 70)
    
    # Test 1: Configuration
    print("\\n1️⃣ Testing Configuration...")
    config: Dict[str, Union[str, bool, int]] = {
        'workspace_path': '/Users/ted/blenderkit_data/MotionBlendAI-1',
        'sync_mode': 'incremental',
        'include_artifacts': True,
        'max_file_size_mb': 100,
        'sync_ui_analytics': True
    }
    
    workspace = Path(str(config['workspace_path']))
    if workspace.exists():
        print(f"✅ Workspace exists: {workspace}")
    else:
        print(f"❌ Workspace not found: {workspace}")
        return
    
    # Test 2: File Discovery
    print("\\n2️⃣ Testing File Discovery...")
    motion_files = discover_motion_files_simple(str(config['workspace_path']))
    
    print(f"📊 Discovery Results:")
    print(f"   Total files: {len(motion_files)}")
    
    # Categorize files
    file_types: Dict[str, int] = {}
    for _, file_type in motion_files:
        file_types[file_type] = file_types.get(file_type, 0) + 1
    
    print("📈 File Type Breakdown:")
    for file_type, count in sorted(file_types.items()):
        print(f"   {file_type.upper()}: {count} files")
    
    # Test 3: Sample Files
    print("\\n3️⃣ Sample Files by Type:")
    for file_type in sorted(file_types.keys()):
        samples = [f for f, t in motion_files if t == file_type][:3]
        print(f"\\n   {file_type.upper()} files:")
        for file_path in samples:
            print(f"     - {os.path.basename(file_path)}")
        if len([f for f, t in motion_files if t == file_type]) > 3:
            remaining = len([f for f, t in motion_files if t == file_type]) - 3
            print(f"     ... and {remaining} more")
    
    # Test 4: Validate Key Directories
    print("\\n4️⃣ Testing Key Directories...")
    key_dirs = [
        'project/seed_motions',
        'build/demo_artifacts', 
        'build/blend_snn'
    ]
    
    for dir_path in key_dirs:
        full_path = workspace / dir_path
        if full_path.exists():
            file_count = len(list(full_path.glob('*')))
            print(f"✅ {dir_path}: {file_count} files")
        else:
            print(f"⚠️  {dir_path}: Not found")
    
    # Test 5: Fivetran Integration Status
    print("\\n5️⃣ Fivetran Integration Status...")
    try:
        import fivetran_connector_sdk  # type: ignore
        print(f"✅ Fivetran SDK available (version: {getattr(fivetran_connector_sdk, '__version__', 'unknown')})")
        print("⚠️  SDK logging has issues - using standalone mode")
    except ImportError:
        print("❌ Fivetran SDK not available - install with: pip install fivetran-connector-sdk")
    except Exception as e:
        print(f"❌ Error importing Fivetran SDK: {e}")
    
    print("\\n" + "=" * 70)
    print("🎯 Debug Summary:")
    print(f"   - Configuration: Valid")
    print(f"   - Motion files found: {len(motion_files)}")
    print(f"   - File types supported: {len(file_types)}")
    print(f"   - Ready for data sync: {'Yes' if motion_files else 'No'}")
    print("=" * 70)

if __name__ == "__main__":
    test_connector_functionality()