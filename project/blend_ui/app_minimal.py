#!/usr/bin/env python3
"""
Minimal Flask App for Blend Strips UI Testing
No Elasticsearch, No Fivetran - Just the UI endpoints
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import numpy as np
import hashlib
import base64
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# WCAG-friendly color palette
DEFAULT_COLORS = [
    "#2FBF71",  # Salsa - Green
    "#F0A202",  # Swing - Amber
    "#2D7DD2",  # Wave - Blue
    "#E63946",  # Funk - Red
    "#9C27B0",  # Jazz - Purple
    "#00897B",  # Ballet - Teal
    "#FF6F00",  # Hip-hop - Orange
]

# Global artifacts storage (in-memory for testing)
ARTIFACTS_STORE = []

def generate_frame_thumbnail(frame_index: int, motion_hash: int, motion_type: str = 'blend') -> str:
    """Generate a placeholder thumbnail data URL for a frame"""
    hue = (motion_hash + frame_index * 137) % 360
    
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="80" height="80" viewBox="0 0 80 80">
        <rect width="80" height="80" fill="hsl({hue}, 60%, 50%)"/>
        <text x="40" y="45" text-anchor="middle" font-size="14" fill="white" font-family="monospace">
            F{frame_index}
        </text>
    </svg>'''
    
    svg_bytes = svg.encode('utf-8')
    b64 = base64.b64encode(svg_bytes).decode('utf-8')
    return f"data:image/svg+xml;base64,{b64}"

def generate_source_motion_data(motion_name: str, motion_hash: int, frames: int, color: str):
    """Generate source motion data for strip visualization"""
    sample_every = 10
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
):
    """Generate blend motion data with segments for strip visualization"""
    sample_every = 10
    sampled_frames = list(range(0, frames, sample_every))
    
    thumbnails = [
        generate_frame_thumbnail(f, motion_hash, 'blend')
        for f in sampled_frames
    ]
    
    color1 = DEFAULT_COLORS[hash(motion1) % len(DEFAULT_COLORS)]
    color2 = DEFAULT_COLORS[hash(motion2) % len(DEFAULT_COLORS)]
    
    segments = []
    
    if transition_start > 0:
        segments.append({
            "fromFrame": 0,
            "toFrame": transition_start - 1,
            "label": motion1,
            "color": color1,
            "alpha": 1.0
        })
    
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

def generate_blend_analysis(motion_hash: int, frames: int, duration: float, weight: float):
    """Generate analysis metrics for a motion blend"""
    np.random.seed(motion_hash)
    
    joint_names = ['Hips', 'LeftWrist', 'RightWrist', 'LeftFoot', 'RightFoot']
    time_points = np.linspace(0, duration, frames)
    
    transition_start = frames // 3
    transition_end = 2 * frames // 3
    
    l2_velocity = {}
    l2_acceleration = {}
    
    for joint in joint_names:
        velocities = []
        for t in range(frames):
            if t < transition_start:
                base_vel = 0.3 + np.random.rand() * 0.2 * (1 - weight)
            elif t < transition_end:
                blend_factor = (t - transition_start) / (transition_end - transition_start)
                base_vel = 0.5 + 0.3 * np.sin(blend_factor * np.pi) + np.random.rand() * 0.3
            else:
                base_vel = 0.3 + np.random.rand() * 0.2 * weight
            
            velocities.append(float(base_vel))
        
        l2_velocity[joint] = velocities
        
        accelerations = [0.0]
        for i in range(1, len(velocities)):
            accel = abs(velocities[i] - velocities[i-1])
            accelerations.append(float(accel))
        
        l2_acceleration[joint] = accelerations
    
    all_velocities = np.concatenate([l2_velocity[j] for j in joint_names])
    all_accelerations = np.concatenate([l2_acceleration[j] for j in joint_names])
    
    transition_velocities = []
    for joint in joint_names:
        transition_velocities.extend(l2_velocity[joint][transition_start:transition_end])
    transition_smoothness = float(np.std(transition_velocities))
    
    global_diversity = float(np.std(all_velocities))
    quality_score = 1.0 - min(transition_smoothness / 0.5, 1.0)
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

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Blend Strips UI (Minimal)",
        "timestamp": datetime.utcnow().isoformat(),
        "artifacts_count": len(ARTIFACTS_STORE)
    })

@app.route('/api/create_blend', methods=['POST', 'OPTIONS'])
@app.route('/api/blend', methods=['POST', 'OPTIONS'])
def create_blend():
    """Create a new motion blend and generate artifact with analysis"""
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response
    
    try:
        logger.info("Creating blend artifact...")
        start_time = time.time()
        
        data = request.get_json()
        motion1 = data.get('motion1', 'walk')
        motion2 = data.get('motion2', 'run')
        weight = data.get('weight', 0.5)
        
        blend_id = f"blend_{int(time.time() * 1000)}"
        timestamp = datetime.utcnow().isoformat()
        
        motion_hash = int(hashlib.md5(f"{motion1}{motion2}".encode()).hexdigest()[:8], 16)
        base_frames = 100 + (motion_hash % 50)
        base_duration = base_frames / 30.0
        
        logger.info(f"Generating analysis for {base_frames} frames...")
        analysis_data = generate_blend_analysis(motion_hash, base_frames, base_duration, weight)
        
        transition_start = analysis_data["transition_window"]["start"]
        transition_end = analysis_data["transition_window"]["end"]
        
        logger.info("Generating source motion data...")
        source1_frames = int(base_frames * 0.8)
        source2_frames = int(base_frames * 0.9)
        
        sources = [
            generate_source_motion_data(motion1, motion_hash, source1_frames, DEFAULT_COLORS[0]),
            generate_source_motion_data(motion2, motion_hash + 1, source2_frames, DEFAULT_COLORS[1])
        ]
        
        logger.info("Generating blend motion with segments...")
        blend_data = generate_blend_motion_data(
            blend_id, motion1, motion2, base_frames, weight,
            motion_hash, transition_start, transition_end
        )
        
        metrics_formatted = {
            "joints": ["pelvis", "lwrist", "rwrist", "lfoot", "rfoot"],
            "l2Velocity": analysis_data["l2_velocity"],
            "l2Acceleration": analysis_data["l2_acceleration"],
            "transitionWindows": [{
                "start": transition_start,
                "end": transition_end
            }]
        }
        
        artifact = {
            "id": blend_id,
            "name": f"{motion1} to {motion2} ({weight:.2f})",
            "type": "motion_blend",
            "status": "completed",
            "createdAt": timestamp,
            "created_at": timestamp,
            "fps": 30,
            "frames": base_frames,
            "sources": sources,
            "blend": blend_data,
            "metrics": metrics_formatted,
            "files": {
                "previewPng": f"/artifacts/{blend_id}_preview.png",
                "metricsJson": f"/artifacts/{blend_id}_metrics.json"
            },
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
        
        ARTIFACTS_STORE.append(artifact)
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Blend created successfully in {elapsed:.2f}s - ID: {blend_id}")
        
        response = jsonify(artifact)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
        
    except Exception as e:
        logger.error(f"❌ Error creating blend: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/artifacts', methods=['GET'])
def list_artifacts():
    """List all artifacts"""
    response = jsonify({
        "artifacts": ARTIFACTS_STORE,
        "count": len(ARTIFACTS_STORE)
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/api/artifact/<artifact_id>/analysis', methods=['GET'])
def get_artifact_analysis(artifact_id):
    """Get analysis data for a specific artifact"""
    for artifact in ARTIFACTS_STORE:
        if artifact['id'] == artifact_id:
            response = jsonify({
                "artifact_id": artifact_id,
                "analysis": artifact.get('analysis', {})
            })
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
    
    return jsonify({"error": "Artifact not found"}), 404

if __name__ == '__main__':
    logger.info("=" * 70)
    logger.info("🚀 Starting Minimal Blend Strips UI Server")
    logger.info("=" * 70)
    logger.info("📡 Server: http://localhost:5000")
    logger.info("🎬 Create blend: POST /api/create_blend")
    logger.info("📋 List artifacts: GET /api/artifacts")
    logger.info("💚 Health check: GET /health")
    logger.info("=" * 70)
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )
