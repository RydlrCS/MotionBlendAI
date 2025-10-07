
import unittest
import json
from unittest.mock import patch, MagicMock
import project.vertex_pipeline.trigger_pipeline_function as trigger_mod

class TestTriggerPipelineFunction(unittest.TestCase):
    @patch('project.vertex_pipeline.trigger_pipeline_function.aiplatform.PipelineJob')
    @patch('project.vertex_pipeline.trigger_pipeline_function.aiplatform.init')
    def test_trigger_vertex_pipeline(self, mock_init: MagicMock, mock_pipeline_job: MagicMock):
        import base64
        payload = json.dumps({
            'pipeline_mode': 'gcs',
            'gcs_data_path': 'gs://test-bucket/data.glb',
            'gcs_model_path': 'gs://test-bucket/model.pth'
        }).encode('utf-8')
        event = {
            'data': base64.b64encode(payload)
        }
        context = None
        mock_job = MagicMock()
        mock_pipeline_job.return_value = mock_job
        mock_job.run.return_value = None
        with patch.dict('os.environ', {
            'PROJECT': 'test-project',
            'LOCATION': 'us-central1',
            'PIPELINE_NAME': 'test-pipeline',
            'PIPELINE_ROOT': 'gs://test-bucket/pipeline-root/',
            'PIPELINE_YAML': 'pipeline.yaml'
        }):
            result = trigger_mod.trigger_vertex_pipeline(event, context)
            if isinstance(result, tuple):
                result = result[0]
            self.assertIn('Triggered Vertex AI pipeline', result['message'])
            mock_init.assert_called_once()
            mock_pipeline_job.assert_called_once()
            mock_job.run.assert_called_once_with(sync=False)

if __name__ == "__main__":
    unittest.main()
