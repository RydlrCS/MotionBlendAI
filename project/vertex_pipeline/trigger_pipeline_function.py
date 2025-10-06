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


from typing import Union, Dict, Any

def trigger_vertex_pipeline(event: Union[Dict[str, Any], str], context=None):
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
    import json
    try:
        # Decode Pub/Sub message
        if isinstance(event, dict) and 'data' in event:
            payload = base64.b64decode(event['data']).decode('utf-8')
            params = json.loads(payload)
        elif isinstance(event, dict):
            params = event  # direct call for testing
        else:
            raise ValueError("Event must be a dictionary.")

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
        return {
            "status": "success",
            "message": f"Triggered Vertex AI pipeline: {pipeline_name}"
        }
    except Exception as e:
        import traceback
        print("Error triggering Vertex AI pipeline:", str(e))
        traceback.print_exc()
        return {
            "status": "error",
            "message": str(e),
            "trace": traceback.format_exc()
        }, 500

    # ...existing code...
# HTTP entry point for manual testing (for local or Cloud Functions HTTP trigger)
def http_trigger(request: Request):
    """
    HTTP trigger for manual testing (expects JSON body with pipeline params).
    """
    try:
        params = request.get_json()
        result = trigger_vertex_pipeline(params)
        from flask import jsonify
        if isinstance(result, tuple):
            # (dict, status_code)
            return jsonify(result[0]), result[1]
        return jsonify(result)
    except Exception as e:
        import traceback
        print("HTTP trigger error:", str(e))
        traceback.print_exc()
        from flask import jsonify
        return jsonify({
            "status": "error",
            "message": str(e),
            "trace": traceback.format_exc()
        }), 500
