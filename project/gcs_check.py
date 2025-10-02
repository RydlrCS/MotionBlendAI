# Script to check Google Cloud Storage bucket availability
from google.cloud import storage
import sys

def check_gcs_bucket(bucket_name):
    try:
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        print(f"Bucket '{bucket_name}' is available and accessible.")
        return True
    except Exception as e:
        print(f"Error accessing bucket '{bucket_name}': {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python gcs_check.py <bucket-name>")
        sys.exit(1)
    check_gcs_bucket(sys.argv[1])
