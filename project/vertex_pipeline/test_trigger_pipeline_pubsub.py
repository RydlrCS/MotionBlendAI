import unittest
from unittest.mock import patch, MagicMock
import project.vertex_pipeline.trigger_pipeline_pubsub as pubsub_mod

class TestTriggerPipelinePubSub(unittest.TestCase):
    @patch('project.vertex_pipeline.trigger_pipeline_pubsub.pubsub_v1.PublisherClient')
    def test_publish_trigger(self, mock_pub_client):
        mock_publisher = MagicMock()
        mock_pub_client.return_value = mock_publisher
        mock_future = MagicMock()
        mock_future.result.return_value = 'message-id-123'
        mock_publisher.publish.return_value = mock_future
        topic = 'projects/test-project/topics/test-topic'
        params = {'pipeline_mode': 'gcs', 'gcs_data_path': 'gs://bucket/data', 'gcs_model_path': 'gs://bucket/model'}
        result = pubsub_mod.publish_trigger(topic, params)
        self.assertEqual(result, 'message-id-123')
        mock_publisher.publish.assert_called_once()
        mock_future.result.assert_called_once()

if __name__ == "__main__":
    unittest.main()
