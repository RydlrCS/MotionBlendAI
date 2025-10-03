"""
Seed script to add sample motion documents into Elasticsearch using SentenceTransformers embeddings.
Run after `docker compose up` when Elasticsearch is available locally.
"""
import time
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import numpy as np

es = Elasticsearch([{"host": "localhost", "port": 9200, "scheme": "http"}])
model = SentenceTransformer('all-MiniLM-L6-v2')

SAMPLES = [
    ("walking into a jump", "walk_jump_1"),
    ("slow run with arm swing", "run_swing_1"),
    ("jump and turn", "jump_turn_1"),
    ("salsa step", "salsa_1"),
    ("swing dance", "swing_1"),
]

def seed():
    if not es.ping():
        print('Elasticsearch not available at localhost:9200')
        return
    # ensure index exists
    index_name = 'motions'
    if not es.indices.exists(index=index_name):
        mapping = {
            "mappings": {
                "properties": {
                    "description": {"type": "text"},
                    "motion_vector": {"type": "dense_vector", "dims": 384}
                }
            }
        }
        es.indices.create(index=index_name, body=mapping)
    # index samples
    for i, (desc, _id) in enumerate(SAMPLES, start=1):
        vec = model.encode(desc)
        doc = {"description": desc, "motion_vector": vec.tolist()}
        es.index(index=index_name, id=str(_id), body=doc)
    es.indices.refresh(index=index_name)
    print('Seeded sample motions')

if __name__ == '__main__':
    seed()
