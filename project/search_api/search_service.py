"""
Search API for hybrid (vector+keyword) search over MoCap motions using Elasticsearch.
Implements rate limiting and embedding as per documentation.
"""
from flask import Flask, request, jsonify
from elasticsearch import Elasticsearch
import time
import threading

app = Flask(__name__)
# Connect to Elasticsearch with scheme parameter
es = Elasticsearch([{"host": "localhost", "port": 9200, "scheme": "http"}])

# Simulated rate limiter for 200GB/s (logic only, not physically enforceable in Python)
RATE_LIMIT_BYTES_PER_SEC = 200 * 1024 ** 3  # 200GB/s
last_request_time = 0
lock = threading.Lock()

def rate_limited():
    global last_request_time
    with lock:
        now = time.time()
        # Simulate a minimum interval between requests (for demo)
        min_interval = 1.0 / (RATE_LIMIT_BYTES_PER_SEC / (1024*1024))  # MB granularity
        if now - last_request_time < min_interval:
            time.sleep(min_interval - (now - last_request_time))
        last_request_time = time.time()

# Dummy embedding function (replace with real model)
def embed_text(text):
    return [0.1] * 128  # Example: 128-dim vector

@app.route('/search', methods=['POST'])
def search():
    rate_limited()
    req = request.get_json()
    text_query = req.get('text_query')
    k = req.get('k', 5)
    q_vec = embed_text(text_query)
    res = es.search(index="motions", body={
        "knn": {"vector_field": {"vector": q_vec, "k": k}},
        "query": {"match": {"description": text_query}}
    })
    return jsonify(res["hits"]["hits"])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
