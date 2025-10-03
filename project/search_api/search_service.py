"""
Search API for hybrid (vector+keyword) search over MoCap motions using Elasticsearch.
Implements rate limiting and embedding as per documentation.
"""
from flask import Flask, request, jsonify
import logging
from typing import Any, Dict, List, Optional
from elasticsearch import Elasticsearch, exceptions as es_exceptions
import time
import threading
from sentence_transformers import SentenceTransformer
import numpy as np

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


import os


def create_es_client():
    # Allow containerized service to talk to the compose service by name.
    host = os.environ.get('ELASTICSEARCH_HOST', 'elasticsearch')
    port = int(os.environ.get('ELASTICSEARCH_PORT', '9200'))
    # Ask for compatibility with ES 8 to avoid media-type mismatches with newer python clients.
    return Elasticsearch(
        [{"host": host, "port": port, "scheme": "http"}],
        headers={"Accept": "application/vnd.elasticsearch+json; compatible-with=8"}
    )


def get_es_client():
    client = app.config.get('ES_CLIENT')
    if client is None:
        client = create_es_client()
        app.config['ES_CLIENT'] = client
    return client

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

# Real embedding model (SentenceTransformers)
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
embed_model = None
try:
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
except Exception:
    embed_model = None

def embed_text(text: Optional[str]) -> List[float]:
    """Return a dense float list embedding for the given text.
    Falls back to a simple hashing vector if the model is unavailable.
    Empty or None text yields a deterministic zero-like vector.
    """
    if not text:
        # deterministic small-value vector for empty input
        rng = np.random.RandomState(0)
        return (rng.rand(384) * 0.001).tolist()
    if embed_model is not None:
        try:
            vec = embed_model.encode(text)
            return vec.tolist() if isinstance(vec, np.ndarray) else list(vec)
        except Exception as ex:
            logger.exception('embedding model failed, falling back to pseudo-embedding')
    # fallback: simple deterministic pseudo-embedding
    rng = np.random.RandomState(abs(hash(text)) % (2**32))
    return rng.rand(384).tolist()


def ensure_index_and_seed():
    """Create `motions` index with a dense_vector mapping and seed example documents.
    This runs at server startup if Elasticsearch is available.
    """
    index_name = "motions"
    es_client = get_es_client()
    if not getattr(es_client, 'ping', lambda: False)():
        return
    try:
        if not es_client.indices.exists(index=index_name):
            # Create index with a dense_vector field `motion_vector` (384 dims for MiniLM)
            mapping = {
                "mappings": {
                    "properties": {
                        "description": {"type": "text"},
                        "motion_vector": {"type": "dense_vector", "dims": 384},
                        "source_file": {"type": "keyword"}
                    }
                }
            }
            # Create with 1 shard and 0 replicas for single-node environments.
            settings = {"settings": {"number_of_shards": 1, "number_of_replicas": 0}}
            body = {**mapping, **settings}
            es_client.indices.create(index=index_name, body=body)
            # Seed a few example documents
            samples = [
                {"description": "walking into a jump", "motion_vector": embed_text("walking into a jump")},
                {"description": "slow run", "motion_vector": embed_text("slow run")},
                {"description": "jump and turn", "motion_vector": embed_text("jump and turn")}
            ]
            for i, doc in enumerate(samples):
                # ensure id is a string
                es_client.index(index=index_name, id=str(i+1), body=doc)
            try:
                # ensure index is fully available in single-node by refreshing
                ref = getattr(es_client.indices, 'refresh', None)
                if callable(ref):
                    ref(index=index_name)
            except Exception:
                pass
        else:
            # if the index exists on a single-node cluster, ensure replicas=0 to avoid unassigned shards
            try:
                es_client.indices.put_settings(index=index_name, body={"index": {"number_of_replicas": 0}})
            except Exception:
                pass
    except es_exceptions.ElasticsearchException:
        # ignore setup errors
        pass


@app.route('/search', methods=['POST'])
def search():
    rate_limited()
    req: Dict[str, Any] = request.get_json() or {}
    text_query: str = req.get('text_query', '')
    k: int = int(req.get('k', 5) or 5)
    q_vec = embed_text(text_query)
    try:
        # Use k-NN vector search if supported
        body = {
            "size": k,
            "query": {
                "knn": {
                    "motion_vector": {
                        "vector": q_vec,
                        "k": k
                    }
                }
            }
        }
        es_client = get_es_client()
        res = es_client.search(index="motions", body=body)
        return jsonify(res.get("hits", {}).get("hits", []))
    except Exception as e:
        # Fall back to a keyword match query if vector search failed
        try:
            es_client = get_es_client()
            res = es_client.search(index="motions", body={"query": {"match": {"description": text_query}}, "size": k})
            return jsonify(res.get("hits", {}).get("hits", []))
        except Exception:
            logger.exception('both vector and keyword search failed')
            return jsonify({"hits": [], "warning": f"Elasticsearch unavailable: {e}"}), 200


@app.route('/health', methods=['GET'])
def health():
    """Simple health endpoint."""
    try:
        # attempt a lightweight info call to ES
        es_client = get_es_client()
        info = es_client.info()
        return jsonify({"status": "ok", "elasticsearch": info}), 200
    except Exception:
        return jsonify({"status": "ok", "elasticsearch": None}), 200


def register_startup():
    """Register ensure_index_and_seed with Flask startup hooks in a
    backwards/forwards-compatible way.
    """
    # Prefer before_first_request if present (older Flask versions).
    if hasattr(app, 'before_first_request'):
        @app.before_first_request
        def _ensure_index_and_seed():
            ensure_index_and_seed()
    # Newer Flask versions expose before_serving; register there if available.
    elif hasattr(app, 'before_serving'):
        @app.before_serving
        def _ensure_index_and_seed_sync():
            # before_serving callbacks may be awaited by the server; keep it simple.
            ensure_index_and_seed()
    else:
        # As a last resort, run the seeding in a background thread so imports don't fail.
        threading.Thread(target=ensure_index_and_seed, daemon=True).start()


# Register startup behavior now (safe across Flask versions)
register_startup()


@app.route('/seed', methods=['POST'])
def seed_endpoint():
    payload = request.get_json() or {}
    folder = payload.get('folder', 'project/seed_motions')
    max_files = payload.get('max_files')
    dry_run = bool(payload.get('dry_run', False))
    try:
        from scripts import seed_motions
        es_client = get_es_client()
        seed_motions.seed_from_folder(folder, es_client=es_client, max_files=max_files, dry_run=dry_run)
        return jsonify({'status': 'seeded', 'folder': folder}), 200
    except Exception as e:
        logger.exception('seeding failed')
        return jsonify({'status': 'error', 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
