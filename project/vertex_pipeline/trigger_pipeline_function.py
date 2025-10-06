"""
Cloud Function to trigger Vertex AI Pipeline on new MoCap data event (Pub/Sub or HTTP).
- Deploy this function to GCP Cloud Functions (Python 3.9+)
- Set environment variables for PROJECT, LOCATION, PIPELINE_NAME, and optionally GCS/BQ paths
- Can be triggered by Pub/Sub (recommended) or HTTP
"""
import os
import base64
from google.cloud import aiplatform
from google.cloud import pubsub_v1
from flask import Request

def trigger_vertex_pipeline(event, context=None):
    """
    Cloud Function entry point for Pub/Sub trigger.
    Expects event with data: {
        "pipeline_mode": "gcs" or "bigquery",
        "gcs_data_path": "gs://...",
        "bq_project": "...",
        "bq_dataset": "...",
        "bq_table": "...",
        "gcs_model_path": "gs://..."
    }
    """
    # Decode Pub/Sub message
    if 'data' in event:
        payload = base64.b64decode(event['data']).decode('utf-8')
        import json
        params = json.loads(payload)
    else:
        params = event  # direct call for testing

    project = os.environ["PROJECT"]
    location = os.environ.get("LOCATION", "us-central1")
    pipeline_name = os.environ.get("PIPELINE_NAME", "mocap-ganimator-training")
    pipeline_root = os.environ.get("PIPELINE_ROOT", "gs://my-bucket/pipeline-root/")
    pipeline_yaml = os.environ.get("PIPELINE_YAML", "pipeline.yaml")

    aiplatform.init(project=project, location=location)
    job = aiplatform.PipelineJob(
        display_name=pipeline_name,
        template_path=pipeline_yaml,
        pipeline_root=pipeline_root,
        parameter_values={
            "pipeline_mode": params.get("pipeline_mode", "gcs"),
            "gcs_data_path": params.get("gcs_data_path", ""),
            "bq_project": params.get("bq_project", ""),
            "bq_dataset": params.get("bq_dataset", ""),
            "bq_table": params.get("bq_table", ""),
            "gcs_model_path": params.get("gcs_model_path", "")
        }
    )
    job.run(sync=False)
    print(f"Triggered Vertex AI pipeline: {pipeline_name}")
    return f"Triggered Vertex AI pipeline: {pipeline_name}"


# HTTP entry point for manual testing (for local or Cloud Functions HTTP trigger)
def http_trigger(request: Request):
    """
    HTTP trigger for manual testing (expects JSON body with pipeline params).
    """
    params = request.get_json()
    return trigger_vertex_pipeline(params)
