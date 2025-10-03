"""
Seed script to add sample motion documents into Elasticsearch using SentenceTransformers embeddings.
Run after `docker compose up` when Elasticsearch is available locally.
"""
from elasticsearch import Elasticsearch
import requests

# Primary ES client (may be incompatible with some client/server combos)
es = Elasticsearch(
    [{"host": "localhost", "port": 9200, "scheme": "http"}],
    headers={"Accept": "application/vnd.elasticsearch+json;compatible-with=8"}
)


class RequestsES:
    """Lightweight ES wrapper using HTTP requests for basic index operations.
    This avoids Python client media-type mismatches in some environments.
    """
    def __init__(self, base_url='http://localhost:9200'):
        self.base = base_url.rstrip('/')

    def ping(self):
        try:
            r = requests.get(f'{self.base}/_cluster/health', timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    class _Indices:
        def __init__(self, base):
            self.base = base

        def exists(self, index):
            r = requests.head(f'{self.base}/{index}')
            return r.status_code == 200

        def create(self, index, body=None):
            requests.put(f'{self.base}/{index}', json=body or {})

        def refresh(self, index):
            requests.post(f'{self.base}/{index}/_refresh')

    @property
    def indices(self):
        return RequestsES._Indices(self.base)

    def index(self, index, id, body):
        requests.put(f'{self.base}/{index}/_doc/{id}', json=body)

# Load embedding model if available (import inside try so script runs without package)
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception:
    model = None

# Supported file extensions for sample motions
SUPPORTED_EXT = {'.fbx', '.bvh', '.trc'}

def text_from_file(path, max_chars=2000):
    """Extract a text snippet from a motion file for embedding.
    For BVH/TRC (text formats) we read the file; for FBX we attempt a text read
    but fall back to filename if unreadable.
    """
    try:
        with open(path, 'r', errors='ignore') as f:
            s = f.read(max_chars)
            return s if s.strip() else None
    except Exception:
        # binary FBX or unreadable — fall back to filename
        return None


def embed_text(text):
    if model is not None:
        try:
            vec = model.encode(text)
            return vec.tolist() if hasattr(vec, 'tolist') else list(vec)
        except Exception as e:
            print(f'Warning: embedding model failed ({e}), falling back to pseudo-embedding')
            # fall through to pseudo-embedding
    # fallback deterministic pseudo-embedding
    import numpy as _np
    rng = _np.random.RandomState(abs(hash(text)) % (2**32))
    return rng.rand(384).tolist()


def seed_from_folder(folder='seed_motions', es_client=None, max_workers=4, max_files=None, dry_run=False):
    """Seed Elasticsearch index from sample motion files in `folder`.

    Args:
        folder: path to the folder containing sample motion files (recursively scanned)
    """
    # determine ES client: provided via DI, or a dry-run dummy, or the module-global `es`
    if es_client is None:
        if dry_run:
            class _DryIndices:
                def exists(self, index):
                    return False
                def create(self, index, body=None):
                    print(f'[dry-run] create index {index}')
                def refresh(self, index):
                    print(f'[dry-run] refresh index {index}')

            class _DryES:
                def __init__(self):
                    self.indices = _DryIndices()
                def ping(self):
                    return True
                def index(self, index, id, body):
                    print(f'[dry-run] would index id={id} file={body.get("source_file")}')

            es_client = _DryES()
        else:
            # prefer the installed Python client, but fall back to requests-based wrapper
            es_client = es
            try:
                ok = es_client.ping()
            except Exception:
                ok = False
            if not ok:
                # fall back to lightweight requests-based client
                try:
                    es_client = RequestsES()
                except Exception:
                    print('Elasticsearch not available at localhost:9200')
                    return

    index_name = 'motions'
    if not es_client.indices.exists(index=index_name):
        mapping = {
            "mappings": {
                "properties": {
                    "description": {"type": "text"},
                    "motion_vector": {"type": "dense_vector", "dims": 384},
                    "source_file": {"type": "keyword"}
                }
            }
        }
        es_client.indices.create(index=index_name, body=mapping)

    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed

    files_to_index = []
    for root, _, files in os.walk(folder):
        for fn in files:
            _, ext = os.path.splitext(fn)
            if ext.lower() not in SUPPORTED_EXT:
                continue
            files_to_index.append(os.path.join(root, fn))

    if max_files is not None:
        files_to_index = files_to_index[:max_files]

    def index_file(path):
        fn = os.path.basename(path)
        desc_text = text_from_file(path)
        if not desc_text:
            # use filename as description fallback
            desc_text = os.path.splitext(fn)[0].replace('_', ' ').replace('-', ' ')
        vec = embed_text(desc_text)
        doc = {"description": desc_text, "motion_vector": vec, "source_file": path}
        doc_id = os.path.relpath(path, folder)
        es_client.index(index=index_name, id=doc_id, body=doc)
        return 1

    count = 0
    # concurrency
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(index_file, p): p for p in files_to_index}
        for fut in as_completed(futures):
            try:
                count += fut.result()
            except Exception as e:
                print(f'Error indexing {futures[fut]}: {e}')

    # refresh index if supported
    try:
        refresh = getattr(es_client.indices, 'refresh', None)
        if callable(refresh):
            refresh(index=index_name)
    except Exception:
        pass
    print(f'Seeded {count} sample motions from {folder}')


def main():
    import argparse
    import os

    p = argparse.ArgumentParser(description='Seed motions into Elasticsearch from a samples folder')
    p.add_argument('--folder', '-f', default=os.environ.get('SAMPLE_DIR', 'seed_motions'),
                   help='Folder containing sample motion files (recursively scanned)')
    p.add_argument('--ext', '-e', nargs='*', default=None,
                   help='Limit to specific extensions (e.g. .bvh .fbx .trc)')
    p.add_argument('--max-files', '-m', type=int, default=None, help='Limit number of files to index')
    p.add_argument('--concurrency', '-c', type=int, default=4, help='Number of parallel workers')
    p.add_argument('--dry-run', action='store_true', help='Do not index, just print what would be done')
    args = p.parse_args()

    # apply ext filter if provided
    global SUPPORTED_EXT
    if args.ext:
        SUPPORTED_EXT = set([x if x.startswith('.') else f'.{x}' for x in args.ext])

    # run seeding
    seed_from_folder(args.folder, max_workers=args.concurrency, max_files=args.max_files, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
