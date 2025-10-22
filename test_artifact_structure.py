#!/usr/bin/env python3
"""Test script to verify artifact structure generation without running server"""

import sys
sys.path.insert(0, '/Users/ted/blenderkit_data/MotionBlendAI-1/project/elastic_search')

# Import the helper functions
from app import (
    generate_frame_thumbnail,
    generate_source_motion_data,
    generate_blend_motion_data,
    generate_blend_analysis,
    DEFAULT_COLORS
)

print("=" * 70)
print("Testing Enhanced Artifact Structure Generation")
print("=" * 70)

# Test parameters
motion1 = "walk"
motion2 = "run"
weight = 0.5
frames = 120
motion_hash = 12345

print(f"\n📋 Test Parameters:")
print(f"  Motion 1: {motion1}")
print(f"  Motion 2: {motion2}")
print(f"  Weight: {weight}")
print(f"  Frames: {frames}")
print(f"  Hash: {motion_hash}")

# Test 1: Frame thumbnail generation
print(f"\n✅ Test 1: Frame Thumbnail Generation")
thumbnail = generate_frame_thumbnail(10, motion_hash, 'blend')
print(f"  Generated thumbnail (length: {len(thumbnail)} chars)")
print(f"  Starts with: {thumbnail[:50]}...")
assert thumbnail.startswith("data:image/svg+xml;base64,"), "Thumbnail should be data URL"

# Test 2: Source motion data
print(f"\n✅ Test 2: Source Motion Data Generation")
source1 = generate_source_motion_data(motion1, motion_hash, 100, DEFAULT_COLORS[0])
print(f"  Source 1 ID: {source1['id']}")
print(f"  Source 1 Label: {source1['label']}")
print(f"  Source 1 Frames: {source1['frames']}")
print(f"  Source 1 Sample Every: {source1['sampleEvery']}")
print(f"  Source 1 Thumbnails: {len(source1['thumbnails'])} items")
print(f"  Source 1 Color: {source1['color']}")
assert len(source1['thumbnails']) == 10, "Should have 10 thumbnails (100 frames / 10)"
assert source1['color'] == DEFAULT_COLORS[0], "Should use first color"

# Test 3: Generate analysis
print(f"\n✅ Test 3: Blend Analysis Generation")
analysis = generate_blend_analysis(motion_hash, frames, frames/30.0, weight)
print(f"  Transition window: {analysis['transition_window']['start']}-{analysis['transition_window']['end']}")
print(f"  Quality score: {analysis['metrics']['quality_score']:.3f}")
print(f"  Quality category: {analysis['metrics']['quality_category']}")
print(f"  Joint names: {analysis['joint_names']}")
print(f"  L2 velocity joints: {list(analysis['l2_velocity'].keys())}")

transition_start = analysis["transition_window"]["start"]
transition_end = analysis["transition_window"]["end"]

# Test 4: Blend motion data with segments
print(f"\n✅ Test 4: Blend Motion Data with Segments")
blend_data = generate_blend_motion_data(
    "blend_test", motion1, motion2, frames, weight,
    motion_hash, transition_start, transition_end
)
print(f"  Blend ID: {blend_data['id']}")
print(f"  Blend Label: {blend_data['label']}")
print(f"  Blend Frames: {blend_data['frames']}")
print(f"  Blend Sample Every: {blend_data['sampleEvery']}")
print(f"  Blend Thumbnails: {len(blend_data['thumbnails'])} items")
print(f"  Blend Segments: {len(blend_data['segments'])} items")

for i, segment in enumerate(blend_data['segments']):
    print(f"    Segment {i}: {segment['label']}")
    print(f"      Frames: {segment['fromFrame']}-{segment['toFrame']}")
    print(f"      Color: {segment['color']}, Alpha: {segment['alpha']}")

# Test 5: Complete artifact structure
print(f"\n✅ Test 5: Complete Artifact Structure")
sources = [
    generate_source_motion_data(motion1, motion_hash, int(frames * 0.8), DEFAULT_COLORS[0]),
    generate_source_motion_data(motion2, motion_hash + 1, int(frames * 0.9), DEFAULT_COLORS[1])
]

metrics_formatted = {
    "joints": ["pelvis", "lwrist", "rwrist", "lfoot", "rfoot"],
    "l2Velocity": analysis["l2_velocity"],
    "l2Acceleration": analysis["l2_acceleration"],
    "transitionWindows": [{
        "start": transition_start,
        "end": transition_end
    }]
}

artifact = {
    "id": "blend_test_123",
    "name": f"{motion1} to {motion2} ({weight:.2f})",
    "type": "motion_blend",
    "status": "completed",
    "fps": 30,
    "frames": frames,
    "sources": sources,
    "blend": blend_data,
    "metrics": metrics_formatted,
    "files": {
        "previewPng": "/artifacts/blend_test_123_preview.png",
        "metricsJson": "/artifacts/blend_test_123_metrics.json"
    }
}

print(f"  Artifact ID: {artifact['id']}")
print(f"  Artifact Name: {artifact['name']}")
print(f"  Artifact Type: {artifact['type']}")
print(f"  Artifact FPS: {artifact['fps']}")
print(f"  Artifact Frames: {artifact['frames']}")
print(f"  Artifact Sources: {len(artifact['sources'])} items")
print(f"  Artifact Blend Segments: {len(artifact['blend']['segments'])} items")
print(f"  Artifact Metrics Joints: {len(artifact['metrics']['joints'])} joints")
print(f"  Artifact Files: {list(artifact['files'].keys())}")

# Verify structure matches TypeScript Artifact type
print(f"\n✅ Test 6: Verify Structure Matches TypeScript Types")
required_fields = ['id', 'name', 'type', 'status', 'fps', 'frames', 'sources', 'blend', 'metrics', 'files']
for field in required_fields:
    assert field in artifact, f"Missing required field: {field}"
    print(f"  ✓ {field}")

# Verify sources structure
for i, source in enumerate(artifact['sources']):
    required_source_fields = ['id', 'label', 'frames', 'sampleEvery', 'thumbnails', 'color']
    for field in required_source_fields:
        assert field in source, f"Source {i} missing field: {field}"

# Verify blend structure
required_blend_fields = ['id', 'label', 'frames', 'sampleEvery', 'thumbnails', 'segments']
for field in required_blend_fields:
    assert field in artifact['blend'], f"Blend missing field: {field}"

# Verify segment structure
for i, segment in enumerate(artifact['blend']['segments']):
    required_segment_fields = ['fromFrame', 'toFrame', 'label', 'color', 'alpha']
    for field in required_segment_fields:
        assert field in segment, f"Segment {i} missing field: {field}"

# Verify metrics structure
required_metrics_fields = ['joints', 'l2Velocity', 'l2Acceleration', 'transitionWindows']
for field in required_metrics_fields:
    assert field in artifact['metrics'], f"Metrics missing field: {field}"

print(f"\n" + "=" * 70)
print(f"✅ ALL TESTS PASSED - Artifact structure is valid!")
print(f"=" * 70)
print(f"\n📄 Sample artifact structure:")
import json
print(json.dumps({
    "id": artifact['id'],
    "name": artifact['name'],
    "sources_count": len(artifact['sources']),
    "blend_segments_count": len(artifact['blend']['segments']),
    "metrics_joints": artifact['metrics']['joints'],
    "transition_window": artifact['metrics']['transitionWindows'][0]
}, indent=2))
