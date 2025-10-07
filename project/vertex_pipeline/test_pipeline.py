import unittest
import os

# kfp is an optional dev dependency; skip these tests if it's not installed
try:
    import kfp  # type: ignore
    KFP_AVAILABLE = True
except Exception:
    KFP_AVAILABLE = False

class TestVertexPipeline(unittest.TestCase):
    def test_pipeline_yaml_exists(self):
        """
        Test that the pipeline.yaml file is generated and not empty.
        """
        yaml_path = os.path.join(os.path.dirname(__file__), 'pipeline.yaml')
        self.assertTrue(os.path.exists(yaml_path), "pipeline.yaml does not exist. Run pipeline compilation.")
        self.assertGreater(os.path.getsize(yaml_path), 100, "pipeline.yaml is too small, likely not compiled correctly.")

    def test_pipeline_signature(self):
        """
        Test that the pipeline has the correct input names for GCS and BigQuery paths.
        """
        if not KFP_AVAILABLE:
            self.skipTest("kfp is not installed in this environment")
        from project.vertex_pipeline.pipeline import mocap_ganimator_pipeline
        input_names = list(mocap_ganimator_pipeline.component_spec.inputs.keys())
        for required in ['pipeline_mode', 'gcs_data_path', 'bq_project', 'bq_dataset', 'bq_table', 'gcs_model_path']:
            self.assertIn(required, input_names)

    def test_pipeline_components(self):
        """
        Test that the pipeline includes the expected components.
        """
        if not KFP_AVAILABLE:
            self.skipTest("kfp is not installed in this environment")
        from project.vertex_pipeline.pipeline import download_mocap_data, train_and_upload_model
        self.assertTrue(callable(download_mocap_data))
        self.assertTrue(callable(train_and_upload_model))

if __name__ == "__main__":
    unittest.main()
