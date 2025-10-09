#!/usr/bin/env python3
"""
MotionBlend AI Elasticsearch API - Quick test version
"""

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Mock data for testing
MOCK_MOTIONS = [
    {
        "id": "motion_001",
        "name": "Walking Forward",
        "vector": [0.12, 0.34, 0.56, 0.78, 0.23, 0.45, 0.67, 0.89],
        "metadata": {
            "category": "locomotion",
            "tags": ["walking", "forward", "basic"],
            "duration": 2.5
        }
    },
    {
        "id": "motion_002", 
        "name": "Dance Hip Hop",
        "vector": [0.89, 0.67, 0.45, 0.23, 0.78, 0.56, 0.34, 0.12],
        "metadata": {
            "category": "dance",
            "tags": ["dance", "hip-hop", "rhythm"],
            "duration": 3.2
        }
    }
]

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "service": "MotionBlend Elasticsearch API",
        "version": "1.0.0",
        "elasticsearch": "mock_mode",
        "motions_loaded": len(MOCK_MOTIONS)
    })

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    query_vector = data.get('vector', [])
    size = data.get('size', 10)
    
    # Simple mock search - return all motions with mock scores
    results = []
    for i, motion in enumerate(MOCK_MOTIONS[:size]):
        results.append({
            **motion,
            "score": 0.95 - (i * 0.1)  # Mock decreasing relevance
        })
    
    return jsonify({
        "results": results,
        "total": len(results),
        "query_type": "vector_similarity"
    })

@app.route('/search/text', methods=['POST'])
def search_text():
    data = request.get_json()
    query = data.get('query', '').lower()
    size = data.get('size', 10)
    
    # Simple text search in mock data
    results = []
    for motion in MOCK_MOTIONS:
        if (query in motion['name'].lower() or 
            any(query in tag.lower() for tag in motion['metadata']['tags']) or
            query in motion['metadata']['category'].lower()):
            results.append(motion)
    
    return jsonify({
        "results": results[:size],
        "total": len(results),
        "query": query,
        "query_type": "text_search"
    })

@app.route('/motions', methods=['GET'])
def get_motions():
    return jsonify(MOCK_MOTIONS)

if __name__ == '__main__':
    print("🚀 Starting MotionBlend AI API (Mock Mode)")
    print(f"📊 Mock motions loaded: {len(MOCK_MOTIONS)}")
    print("🌐 Server starting on http://127.0.0.1:5004")
    app.run(debug=True, host='127.0.0.1', port=5004)