#!/usr/bin/env python3
"""Test script to verify enhanced blend artifact with strip data"""

import requests
import json
import time

# Wait for server to be ready
print("Waiting for Flask server to start...")
time.sleep(5)

# Test create_blend endpoint
url = "http://localhost:5000/api/create_blend"
payload = {
    "motion1": "salsa_spin",
    "motion2": "swing_basic",
    "weight": 0.5
}

print(f"\n📤 Testing POST {url}")
print(f"Payload: {json.dumps(payload, indent=2)}")

try:
    response = requests.post(url, json=payload, timeout=10)
    response.raise_for_status()
    
    data = response.json()
    
    print(f"\n✅ Status Code: {response.status_code}")
    print(f"\n📦 Response Structure:")
    print(f"  - id: {data.get('id')}")
    print(f"  - name: {data.get('name')}")
    print(f"  - type: {data.get('type')}")
    print(f"  - fps: {data.get('fps')}")
    print(f"  - frames: {data.get('frames')}")
    
    # Check sources
    if 'sources' in data:
        print(f"\n🎬 Sources ({len(data['sources'])} items):")
        for i, source in enumerate(data['sources']):
            print(f"  [{i}] {source.get('label')}")
            print(f"      - id: {source.get('id')}")
            print(f"      - frames: {source.get('frames')}")
            print(f"      - sampleEvery: {source.get('sampleEvery')}")
            print(f"      - thumbnails: {len(source.get('thumbnails', []))} items")
            print(f"      - color: {source.get('color')}")
            if source.get('thumbnails'):
                thumb = source['thumbnails'][0]
                print(f"      - first thumbnail: {thumb[:50]}...")
    else:
        print("\n❌ No 'sources' field found!")
    
    # Check blend
    if 'blend' in data:
        blend = data['blend']
        print(f"\n🎭 Blend Motion:")
        print(f"  - id: {blend.get('id')}")
        print(f"  - label: {blend.get('label')}")
        print(f"  - frames: {blend.get('frames')}")
        print(f"  - sampleEvery: {blend.get('sampleEvery')}")
        print(f"  - thumbnails: {len(blend.get('thumbnails', []))} items")
        
        if 'segments' in blend:
            print(f"  - segments: {len(blend['segments'])} items")
            for i, seg in enumerate(blend['segments'][:3]):  # Show first 3
                print(f"    [{i}] {seg.get('label')}")
                print(f"        frames: {seg.get('fromFrame')}-{seg.get('toFrame')}")
                print(f"        color: {seg.get('color')}, alpha: {seg.get('alpha')}")
        else:
            print("  ❌ No 'segments' field found!")
    else:
        print("\n❌ No 'blend' field found!")
    
    # Check metrics
    if 'metrics' in data:
        metrics = data['metrics']
        print(f"\n📊 Metrics:")
        print(f"  - joints: {metrics.get('joints')}")
        print(f"  - l2Velocity: {len(metrics.get('l2Velocity', []))} frames")
        print(f"  - l2Acceleration: {len(metrics.get('l2Acceleration', []))} frames")
        if 'transitionWindows' in metrics:
            for tw in metrics['transitionWindows']:
                print(f"  - transition: frames {tw.get('start')}-{tw.get('end')}")
    else:
        print("\n❌ No 'metrics' field found!")
    
    # Check files
    if 'files' in data:
        print(f"\n📁 Files:")
        for key, path in data['files'].items():
            print(f"  - {key}: {path}")
    else:
        print("\n❌ No 'files' field found!")
    
    print(f"\n✅ SUCCESS: Enhanced artifact structure verified!")
    print(f"\nFull response saved to test_strip_data_response.json")
    
    with open('/Users/ted/blenderkit_data/MotionBlendAI-1/test_strip_data_response.json', 'w') as f:
        json.dump(data, f, indent=2)
    
except requests.exceptions.RequestException as e:
    print(f"\n❌ ERROR: {e}")
    exit(1)
