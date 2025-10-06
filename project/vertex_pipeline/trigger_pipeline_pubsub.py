"""
Pub/Sub publisher utility for triggering the Vertex AI pipeline from anywhere (e.g., after BigQuery insert).
- Use this script to publish a message to the trigger topic.
- Example usage: python3 trigger_pipeline_pubsub.py --topic projects/PROJECT_ID/topics/TRIGGER_TOPIC --pipeline_mode gcs --gcs_data_path gs://... --gcs_model_path gs://...
"""
import argparse
import json
import base64
from google.cloud import pubsub_v1

def publish_trigger(topic, params):
    publisher = pubsub_v1.PublisherClient()
    data = json.dumps(params).encode('utf-8')
    future = publisher.publish(topic, data)
    print(f"Published trigger to {topic}: {params}")
    return future.result()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--topic', required=True, help='Pub/Sub topic (projects/PROJECT_ID/topics/TRIGGER_TOPIC)')
    parser.add_argument('--pipeline_mode', default='gcs', help='Pipeline mode: gcs or bigquery')
    parser.add_argument('--gcs_data_path', default='', help='GCS path to MoCap data')
    parser.add_argument('--bq_project', default='', help='BigQuery project (if using bigquery mode)')
    parser.add_argument('--bq_dataset', default='', help='BigQuery dataset (if using bigquery mode)')
    parser.add_argument('--bq_table', default='', help='BigQuery table (if using bigquery mode)')
    parser.add_argument('--gcs_model_path', default='', help='GCS path to output model')
    args = parser.parse_args()
    params = {
        'pipeline_mode': args.pipeline_mode,
        'gcs_data_path': args.gcs_data_path,
        'bq_project': args.bq_project,
        'bq_dataset': args.bq_dataset,
        'bq_table': args.bq_table,
        'gcs_model_path': args.gcs_model_path
    }
    publish_trigger(args.topic, params)
