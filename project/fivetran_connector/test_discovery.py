#!/usr/bin/env python3
"""
Motion File Discovery Test Script

Simple standalone test to verify motion file discovery
without Fivetran SDK dependencies.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple


def discover_motion_files(workspace_path: str) -> List[Tuple[str, str]]:
    """
    Discover all motion capture files across the workspace.
    
    Comprehensive file discovery supporting multiple motion capture formats.
    
    Args:
        workspace_path: Root path of MotionBlendAI workspace
        
    Returns:
        List of tuples (file_path, file_type) for all discovered motion files
    """
    motion_files: List[Tuple[str, str]] = []
    workspace = Path(workspace_path)
    
    # Define search paths with priority order
    search_paths = [
        workspace / 'project' / 'seed_motions',     # Primary motion library
        workspace / 'build' / 'demo_artifacts',     # Generated demonstrations
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
    
    print(f"🔍 Discovering motion files in workspace: {workspace_path}")
    
    for search_path in search_paths:
        if not search_path.exists():
            print(f"⚠️  Search path does not exist: {search_path}")
            continue
            
        print(f"📁 Scanning directory: {search_path.relative_to(workspace)}")
        
        try:
            # Recursively search for motion files
            for file_path in search_path.rglob('*'):
                if file_path.is_file():
                    file_ext = file_path.suffix.lower()
                    
                    if file_ext in file_type_mapping:
                        file_type = file_type_mapping[file_ext]
                        motion_files.append((str(file_path), file_type))
                        print(f"  ✅ Found {file_type.upper()}: {file_path.relative_to(workspace)}")
                        
        except Exception as e:
            print(f"❌ Error scanning {search_path}: {e}")
            continue
    
    return motion_files


def main():
    """Main test function."""
    print("=" * 80)
    print("MotionBlend AI - Motion File Discovery Test")
    print("=" * 80)
    
    workspace_path = "/Users/ted/blenderkit_data/MotionBlendAI-1"
    
    # Test motion file discovery
    motion_files = discover_motion_files(workspace_path)
    
    print(f"\n📊 Discovery Results:")
    print(f"   Total files found: {len(motion_files)}")
    
    # Categorize by type
    file_types: Dict[str, int] = {}
    for file_path, file_type in motion_files:
        file_types[file_type] = file_types.get(file_type, 0) + 1
    
    print(f"\n📈 File Type Breakdown:")
    for file_type, count in sorted(file_types.items()):
        print(f"   {file_type.upper()}: {count} files")
    
    # Show sample files from each type
    print(f"\n📄 Sample Files by Type:")
    for file_type in sorted(file_types.keys()):
        sample_files = [f for f, t in motion_files if t == file_type][:3]
        print(f"\n   {file_type.upper()} files:")
        for file_path in sample_files:
            print(f"     - {os.path.basename(file_path)}")
        if len([f for f, t in motion_files if t == file_type]) > 3:
            remaining = len([f for f, t in motion_files if t == file_type]) - 3
            print(f"     ... and {remaining} more files")
    
    print(f"\n✅ Motion file discovery completed successfully!")
    print(f"🔄 Ready for Fivetran sync: {len(motion_files)} files indexed")
    print("=" * 80)


if __name__ == "__main__":
    main()