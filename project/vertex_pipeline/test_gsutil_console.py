import subprocess
import os

def test_gsutil_cp_pipeline_yaml():
    """
    Test that gsutil cp uploads pipeline.yaml to the correct GCS location and returns success.
    """
    local_path = "/Users/ted/blenderkit_data/MotionBlendAI-1/project/vertex_pipeline/pipeline.yaml"
    gcs_path = "gs://motionblend-ai/pipeline-root/pipeline.yaml"
    # Run gsutil cp command
    result = subprocess.run([
        "gsutil", "cp", local_path, gcs_path
    ], capture_output=True, text=True)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    assert result.returncode == 0, f"gsutil cp failed: {result.stderr}"
    assert "Operation completed" in result.stdout or "Operation completed" in result.stderr, "Upload did not complete successfully."
    print("gsutil cp pipeline.yaml test passed.")

if __name__ == "__main__":
    test_gsutil_cp_pipeline_yaml()
