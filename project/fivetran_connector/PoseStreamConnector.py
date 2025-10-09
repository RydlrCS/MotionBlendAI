
import os
import time
from typing import Dict, Any, List, Optional, Union

# --- PoseStreamConnector for Moverse MoCap Ingestion ---
# This file implements a Fivetran-compatible connector for ingesting motion-capture data
# from the Moverse API or from local BVH, FBX, TRC files. It is designed for clarity and
# hackathon demonstration, with detailed comments for each step.

# --- Dummy base class for local testing (remove if using real Fivetran SDK) ---

from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPIError

class DummyConnector:
    """A base class for Fivetran-style ingestion with real BigQuery integration."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.bq_client = bigquery.Client()
        self.bq_table = config.get('bigquery_table')

    def load(self, record: Dict[str, Any]):
        """
        Insert a record into BigQuery. Assumes table exists and schema matches record keys.
        """
        try:
            errors = self.bq_client.insert_rows_json(self.bq_table, [record])
            if errors:
                print(f"[BigQuery] Insert errors: {errors}")
            else:
                print(f"[BigQuery] Inserted record at {record.get('timestamp')}")
        except GoogleAPIError as e:
            print(f"[BigQuery] API error: {e}")
        except Exception as e:
            print(f"[BigQuery] Unexpected error: {e}")

# --- Main Connector Class ---
class PoseStreamConnector(DummyConnector):
    """
    Streams MoCap frames from Moverse API or local files to BigQuery.
    - Real-time mode: fetches frames from Moverse API.
    - Batch mode: parses BVH, FBX, TRC files from a folder.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # API and mode configuration
        self.api_url = config.get('moverse_api_url')
        self.api_key = config.get('moverse_api_key')
        self.mode = config.get('mode', 'stream')  # 'stream' or 'batch'
        self.file_folder = config.get('file_folder', None)  # For batch mode
        self.bigquery_table = config.get('bigquery_table')

    def connect(self):
        """
        Entry point for the connector. Chooses streaming or batch mode.
        """
        if self.mode == 'stream':
            self._stream_from_api()
        elif self.mode == 'batch':
            self._batch_from_files()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _stream_from_api(self):
        """
        Connect to Moverse API and stream frames in real time.
        Each frame is preprocessed and sent to BigQuery.
        """
        import requests
        headers = {'Authorization': f'Bearer {self.api_key}'}
        print("[INFO] Starting real-time streaming from Moverse API...")
        for i in range(3):  # Limit to 3 frames for demo/testing
            try:
                # Simulate API call (replace with real endpoint)
                resp = requests.get(f"{self.api_url}/realtime_frame", headers=headers, timeout=10)
                if resp.status_code == 200:
                    frame = resp.json()
                    processed = preprocess_frame(frame)
                    self.load(processed)
                else:
                    print(f"[WARN] API status {resp.status_code}, retrying...")
                    time.sleep(1)
            except Exception as e:
                print(f"[Moverse] Error fetching frame: {e}")
                time.sleep(2)

    def _batch_from_files(self):
        """
        Ingest all BVH, FBX, TRC files from a folder (for batch import).
        Each file is parsed, preprocessed, and sent to BigQuery.
        """
        supported_ext = {'.bvh', '.fbx', '.trc'}
        print(f"[INFO] Scanning folder {self.file_folder} for motion files...")
        for root, _, files in os.walk(self.file_folder):
            for fn in files:
                ext = os.path.splitext(fn)[1].lower()
                if ext not in supported_ext:
                    continue
                file_path = os.path.join(root, fn)
                frame = parse_motion_file(file_path, ext)
                if frame:
                    processed = preprocess_frame(frame)
                    self.load(processed)

# --- Preprocessing and Parsing Utilities ---
def preprocess_frame(frame: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize and filter frame data for BigQuery schema.
    Ensures keys: timestamp, joints, meta.
    """
    out = {
        'timestamp': frame.get('timestamp', time.time()),
        'joints': frame.get('joints', []),
        'meta': frame.get('meta', {}),
    }
    return out

def parse_motion_file(file_path: str, ext: str) -> Optional[Dict[str, Any]]:
    """
    Parse a motion file (BVH, FBX, TRC) and extract a frame or sequence.
    Returns a dict or None if parse fails.
    """
    try:
        if ext == '.bvh':
            return parse_bvh(file_path)
        elif ext == '.trc':
            return parse_trc(file_path)
        elif ext == '.fbx':
            return parse_fbx(file_path)
    except Exception as e:
        print(f"[Moverse] Failed to parse {file_path}: {e}")
    return None

from typing import Optional, Dict, Any

def parse_bvh(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Minimal BVH parser: extracts first frame of motion data.
    """
    with open(file_path, 'r', errors='ignore') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if 'Frame Time' in line:
            if i+1 < len(lines):
                data = lines[i+1].strip().split()
                joints = [float(x) for x in data if x.replace('.', '', 1).isdigit() or x.replace('.', '', 1).replace('-', '', 1).isdigit()]
                return {
                    'timestamp': time.time(),
                    'joints': joints,
                    'meta': {
                        'source': 'bvh',
                        'file': file_path
                    }
                }
    return None

def parse_trc(file_path):
    """
    Minimal TRC parser: extracts first data row after header.
    """
    with open(file_path, 'r', errors='ignore') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.strip().startswith('Frame#') and i+1 < len(lines):
            data = lines[i+1].strip().split()
            joints = [float(x) for x in data[2:] if x.replace('.', '', 1).isdigit() or x.replace('.', '', 1).replace('-', '', 1).isdigit()]
            return {'timestamp': time.time(), 'joints': joints, 'meta': {'source': 'trc', 'file': file_path}}
    return None

def parse_fbx(file_path: str) -> Dict[str, Union[float, List[float], Dict[str, str]]]:
    """
    FBX parser stub: just records file info (real parser needed for production).
    """
    return {
        'timestamp': float(time.time()),  # float type
        'joints': [],  # List[float] type (empty list for now)
        'meta': {  # Dict[str, str] type
            'source': 'fbx',
            'file': str(file_path)
        }
    }

# --- Test Harness for Local Testing ---
if __name__ == '__main__':

    # Example config for batch mode (test with seed_motions FBX files)
    config = {
        'mode': 'batch',
        'file_folder': 'project/seed_motions',  # Use the actual seed_motions folder
        'bigquery_table': 'myproject.dataset.motions',
    }
    connector = PoseStreamConnector(config)
    connector.connect()

    # Example config for stream mode (test with dummy API)
    # config = {
    #     'mode': 'stream',
    #     'moverse_api_url': 'http://localhost:5000',
    #     'moverse_api_key': 'testkey',
    #     'bigquery_table': 'myproject.dataset.motions',
    # }
    # connector = PoseStreamConnector(config)
    # connector.connect()
