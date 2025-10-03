import kfp
from google.cloud import aiplatform
from google_cloud_pipeline_components.v1.dataset import ImageDatasetCreateOp
from google_cloud_pipeline_components.v1.automl.training_job import AutoMLImageTrainingJobRunOp
from google_cloud_pipeline_components.v1.endpoint import EndpointCreateOp, ModelDeployOp
import os
from blending.ganimator_blender import GANimatorBlender

@kfp.dsl.pipeline(
    name="automl-image-training",
    pipeline_root="gs://my-bucket/pipeline-root/") # type: ignore
def pipeline(project_id: str):
    """
    Vertex AI pipeline for AutoML image training and deployment.
    Steps:
      1. Create an AutoML dataset from CSV on GCS
      2. Train an AutoML model using the dataset
      3. Create an endpoint and deploy the trained model
    Args:
        project_id (str): GCP project ID
    """
    # 1) Create an AutoML dataset from CSV on GCS
    ds_op = ImageDatasetCreateOp(
        project=project_id,
        display_name="flowers-dataset",
        gcs_source="gs://my-bucket/data.csv",
        import_schema_uri=aiplatform.schema.dataset.ioformat.image.single_label_classification,
    )
    # 2) Train an AutoML model using the dataset
    training_job = AutoMLImageTrainingJobRunOp(
        project=project_id,
        display_name="flowers-training-job",
        prediction_type="classification",
        model_type="CLOUD",
        dataset=ds_op.outputs["dataset"],
        model_display_name="flowers-model",
    )
    # 3) Create an endpoint and deploy the trained model
    endpoint_op = EndpointCreateOp(project=project_id, display_name="flowers-endpoint")
    ModelDeployOp(
        model=training_job.outputs["model"],
        endpoint=endpoint_op.outputs["endpoint"],
        deployed_model_display_name="flowers-deployment",
        automatic_resources_min_replica_count=1,
        automatic_resources_max_replica_count=1,
    )

def train_and_blend_pipeline(project_id, bucket_name, model_path):
    """
    Integrate the blending module into the pipeline.
    This function demonstrates training and blending using the GANimatorBlender.
    Args:
        project_id (str): GCP project ID.
        bucket_name (str): GCS bucket name for model storage.
        model_path (str): Path to the model in GCS.
    """
    # Initialize the blender with the model from GCS
    blender = GANimatorBlender(model_path=model_path, bucket_name=bucket_name)

    # Example motion sequences (dummy data)
    seq1 = torch.randn(1, 10)  # Replace with actual motion sequence
    seq2 = torch.randn(1, 10)  # Replace with actual motion sequence
    mix_ratio = 0.5  # Blend factor

    # Perform blending
    blended_motion = blender.blend(seq1, seq2, mix_ratio)
    print("Blended Motion:", blended_motion)

# Example usage
if __name__ == "__main__":
    PROJECT_ID = os.getenv("GCP_PROJECT_ID", "your-project-id")
    BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "your-bucket-name")
    MODEL_PATH = os.getenv("MODEL_PATH", "path/to/ganimator_model.pth")

    train_and_blend_pipeline(PROJECT_ID, BUCKET_NAME, MODEL_PATH)

    # Compile the pipeline to a YAML for Vertex AI
    kfp.compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path='pipeline.yaml')
