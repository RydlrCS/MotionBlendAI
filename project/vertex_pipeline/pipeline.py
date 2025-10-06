
import kfp
from kfp.v2.dsl import component, pipeline
import os
from google.cloud import storage

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
def mocap_ganimator_pipeline(gcs_data_path: str, gcs_model_path: str):
    """
    Pipeline steps:
      1. Download MoCap data from GCS
      2. Train SPADE-GANimator and upload model to GCS
    Args:
        gcs_data_path: GCS path to MoCap data file (e.g. .glb, .trc, or .npy)
        gcs_model_path: GCS path to upload trained model
    """
    local_data_path = "/tmp/mocap_data_file"
    model_local_path = "models/ganimator_spade.pth"
    data_prep = download_mocap_data(gcs_data_path=gcs_data_path, local_data_path=local_data_path)
    train_and_upload_model(model_local_path=model_local_path, gcs_model_path=gcs_model_path, local_data_path=data_prep.output)


# Example usage

if __name__ == "__main__":
    GCS_DATA_PATH = os.getenv("GCS_DATA_PATH", "gs://your-bucket/mocap_data/example.glb")
    GCS_MODEL_PATH = os.getenv("GCS_MODEL_PATH", "gs://your-bucket/models/ganimator_spade.pth")
    # Compile the pipeline to a YAML for Vertex AI
    kfp.compiler.Compiler().compile(
        pipeline_func=mocap_ganimator_pipeline,
        package_path='pipeline.yaml')
