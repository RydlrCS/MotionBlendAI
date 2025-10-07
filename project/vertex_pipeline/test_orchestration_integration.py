import os
import json
import base64
import unittest
from unittest.mock import patch, MagicMock

import project.vertex_pipeline.trigger_pipeline_function as trigger_mod


class TestOrchestrationIntegration(unittest.TestCase):
    @patch('project.vertex_pipeline.trigger_pipeline_function.aiplatform.PipelineJob')
    @patch('project.vertex_pipeline.trigger_pipeline_function.aiplatform.init')
    def test_trigger_pipeline_runs(self, mock_init: MagicMock, mock_pipeline_job: MagicMock):
        payload = json.dumps({
            'pipeline_mode': 'gcs',
            'gcs_data_path': 'gs://test-bucket/data.glb',
            'gcs_model_path': 'gs://test-bucket/model.pth'
        }).encode('utf-8')
        event = {'data': base64.b64encode(payload)}

        mock_job = MagicMock()
        mock_pipeline_job.return_value = mock_job
        mock_job.run.return_value = None

        with patch.dict(os.environ, {
            'PROJECT': 'integration-project',
            'LOCATION': 'us-central1',
            'PIPELINE_NAME': 'integration-pipeline',
            'PIPELINE_ROOT': 'gs://integration-bucket/pipeline-root/',
            'PIPELINE_YAML': 'pipeline.yaml'
        }):
            result = trigger_mod.trigger_vertex_pipeline(event)

            # Unpack tuple return if present
            if isinstance(result, tuple):
                result = result[0]

            # Basic success response
            assert result.get('status') == 'success'
            assert 'Triggered Vertex AI pipeline' in result.get('message', '')

            # Ensure aiplatform was initialized and PipelineJob created
            mock_init.assert_called_once()
            mock_pipeline_job.assert_called_once()

            # Check that PipelineJob was constructed with expected args
            _, called_kwargs = mock_pipeline_job.call_args
            assert called_kwargs.get('display_name') == 'integration-pipeline'
            assert called_kwargs.get('template_path') == 'pipeline.yaml'

            # Ensure the job was run asynchronously
            mock_job.run.assert_called_once_with(sync=False)


if __name__ == '__main__':
    unittest.main()
