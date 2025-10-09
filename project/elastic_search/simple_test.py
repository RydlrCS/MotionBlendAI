#!/usr/bin/env python3
"""
Simple test server for debugging
"""

from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "service": "MotionBlend Elasticsearch API",
        "version": "1.0.0"
    })

@app.route('/test', methods=['GET'])  
def test():
    return jsonify({"message": "Server is working!"})

if __name__ == '__main__':
    print("🚀 Starting simple test server...")
    app.run(debug=True, host='127.0.0.1', port=5003)