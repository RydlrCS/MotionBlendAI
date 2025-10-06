import requests
import time
import os

def test_vertex_pipeline_orchestration():
    """
    Integration test: Trigger the Cloud Function and check for successful pipeline job creation.
    """
    # Set up test payload
    payload = {
        "pipeline_mode": "gcs",
        "gcs_data_path": "gs://motionblend-ai/data",
        "gcs_model_path": "gs://motionblend-ai/model"
    }
    url = os.environ.get("CLOUD_FUNCTION_URL", "https://us-central1-motionblend-ai.cloudfunctions.net/trigger_vertex_pipeline")
    response = requests.post(url, json=payload)
    print("Raw response:", response.text)
    try:
        result = response.json()
        assert result.get("status") == "success", f"Pipeline trigger failed: {result}"
        print("Vertex AI pipeline orchestration test passed.")
    except Exception as e:
        print("Failed to decode JSON response:", e)
        print("Response status code:", response.status_code)
        print("Response text:", response.text)

if __name__ == "__main__":
    test_vertex_pipeline_orchestration()
