"""
MotionBlendAI Vertex AI Pipeline
--------------------------------
This pipeline orchestrates MoCap data ingestion, SPADE-GANimator training, and model deployment for the hackathon.

Features:
- Ingest MoCap data from GCS (default) or BigQuery (optional)
- Train SPADE-GANimator on the data
- Upload trained model to GCS

Configuration:
- Set GCS_DATA_PATH (gs://...) for GCS file input
- Set BQ_PROJECT, BQ_DATASET, BQ_TABLE for BigQuery input (optional)
- Set GCS_MODEL_PATH for model output
- Set PIPELINE_MODE to 'gcs' (default) or 'bigquery'

Usage:
    python3 pipeline.py  # Compiles pipeline.yaml for Vertex AI
    # See README.md for full deployment instructions
"""

import kfp
from kfp.v2.dsl import component, pipeline
import os
from google.cloud import storage
from google.cloud import bigquery

# --- Custom component to train SPADE-GANimator and upload model to GCS ---


# Data prep step: download MoCap data from GCS to local path
@component(base_image="python:3.9")
def download_mocap_data(gcs_data_path: str, local_data_path: str) -> str:
    """
    Downloads a file from GCS to a local path for use in the pipeline.
    """
    from google.cloud import storage
    import os
    client = storage.Client()
    bucket_name, blob_path = gcs_data_path.replace("gs://", "").split("/", 1)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    os.makedirs(os.path.dirname(local_data_path), exist_ok=True)
    blob.download_to_filename(local_data_path)
    print(f"Downloaded {gcs_data_path} to {local_data_path}")
    return local_data_path

# Data prep step: download MoCap data from BigQuery to local path (optional)
@component(base_image="python:3.9")
def download_mocap_from_bigquery(project: str, dataset: str, table: str, local_data_path: str) -> str:
    """
    Downloads MoCap data from a BigQuery table to a local file (as JSONL).
    """
    from google.cloud import bigquery
    import json
    import os
    client = bigquery.Client(project=project)
    query = f"SELECT * FROM `{project}.{dataset}.{table}`"
    job = client.query(query)
    rows = list(job)
    os.makedirs(os.path.dirname(local_data_path), exist_ok=True)
    with open(local_data_path, 'w') as f:
        for row in rows:
            f.write(json.dumps(dict(row)) + '\n')
    print(f"Downloaded {len(rows)} rows from BigQuery to {local_data_path}")
    return local_data_path

# Training step: run train.py and upload model to GCS
@component(base_image="python:3.9")
def train_and_upload_model(model_local_path: str, gcs_model_path: str, local_data_path: str) -> str:
    import subprocess
    import os
    # Run the training script, passing the local data path as an env var
    env = os.environ.copy()
    env["MOCAP_DATA_PATH"] = local_data_path
    subprocess.run(["python3", "project/ganimator/train.py"], check=True, env=env)
    # Upload the model to GCS
    from google.cloud import storage
    client = storage.Client()
    bucket_name, blob_path = gcs_model_path.replace("gs://", "").split("/", 1)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(model_local_path)
    print(f"Uploaded model to {gcs_model_path}")
    return gcs_model_path


# --- MoCap SPADE-GANimator pipeline ---


@pipeline(
    name="mocap-ganimator-training",
    pipeline_root="gs://my-bucket/pipeline-root/"
)
def mocap_ganimator_pipeline(
    pipeline_mode: str = 'gcs',
    gcs_data_path: str = '',
    bq_project: str = '',
    bq_dataset: str = '',
    bq_table: str = '',
    gcs_model_path: str = ''
):
    """
    Pipeline steps:
      1. Download MoCap data from GCS or BigQuery
      2. Train SPADE-GANimator and upload model to GCS
    Args:
        pipeline_mode: 'gcs' (default) or 'bigquery'
        gcs_data_path: GCS path to MoCap data file (e.g. .glb, .trc, or .npy)
        bq_project, bq_dataset, bq_table: BigQuery source (if pipeline_mode='bigquery')
        gcs_model_path: GCS path to upload trained model
    """
    local_data_path = "/tmp/mocap_data_file"
    model_local_path = "models/ganimator_spade.pth"
    if pipeline_mode == 'bigquery':
        data_prep = download_mocap_from_bigquery(
            project=bq_project,
            dataset=bq_dataset,
            table=bq_table,
            local_data_path=local_data_path)
    else:
        data_prep = download_mocap_data(
            gcs_data_path=gcs_data_path,
            local_data_path=local_data_path)
    train_and_upload_model(
        model_local_path=model_local_path,
        gcs_model_path=gcs_model_path,
        local_data_path=data_prep.output)


# Example usage

if __name__ == "__main__":
    PIPELINE_MODE = os.getenv("PIPELINE_MODE", "gcs")
    GCS_DATA_PATH = os.getenv("GCS_DATA_PATH", "gs://your-bucket/mocap_data/example.glb")
    BQ_PROJECT = os.getenv("BQ_PROJECT", "")
    BQ_DATASET = os.getenv("BQ_DATASET", "")
    BQ_TABLE = os.getenv("BQ_TABLE", "")
    GCS_MODEL_PATH = os.getenv("GCS_MODEL_PATH", "gs://your-bucket/models/ganimator_spade.pth")
    # Compile the pipeline to a YAML for Vertex AI
    kfp.compiler.Compiler().compile(
        pipeline_func=mocap_ganimator_pipeline,
        package_path='pipeline.yaml')
