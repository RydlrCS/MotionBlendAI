import kfp
from google.cloud import aiplatform
from google_cloud_pipeline_components.v1.dataset import ImageDatasetCreateOp
from google_cloud_pipeline_components.v1.automl.training_job import AutoMLImageTrainingJobRunOp
from google_cloud_pipeline_components.v1.endpoint import EndpointCreateOp, ModelDeployOp

@kfp.dsl.pipeline(
    name="automl-image-training",
    pipeline_root="gs://my-bucket/pipeline-root/")
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

if __name__ == '__main__':
    # Compile the pipeline to a YAML for Vertex AI
    kfp.compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path='pipeline.yaml')
