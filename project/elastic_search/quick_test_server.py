#!/usr/bin/env python3
"""
Quick test server with immediate response - no external dependencies
"""

from flask import Flask, jsonify, request
import json

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "service": "MotionBlend Test API",
        "version": "1.0.0",
        "mode": "quick_test"
    })

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json() or {}
    return jsonify({
        "results": [
            {"name": "Test Motion 1", "score": 0.95},
            {"name": "Test Motion 2", "score": 0.87}
        ],
        "total": 2,
        "query_type": "mock"
    })

@app.route('/search/text', methods=['POST'])
def search_text():
    data = request.get_json() or {}
    query = data.get('query', 'test')
    return jsonify({
        "results": [{"name": f"Motion matching '{query}'", "score": 1.0}],
        "total": 1,
        "query": query
    })

@app.route('/motions', methods=['GET'])
def get_motions():
    return jsonify([
        {"id": "1", "name": "Test Motion", "category": "test"}
    ])

@app.route('/index/bulk', methods=['POST'])
def bulk_index():
    data = request.get_json() or {}
    docs = data.get('documents', [])
    return jsonify({
        "success": True,
        "indexed": len(docs),
        "errors": 0
    })

if __name__ == '__main__':
    print("🚀 Quick Test Server Starting...")
    app.run(host='127.0.0.1', port=5005, debug=False)