from fivetran_sdk import Connector

class PoseStreamConnector(Connector):
    def connect(self):
        while True:
            frame = capture_next_frame()       # grab latest MoCap frame (3D coordinates, timestamp, etc.)
            processed = preprocess(frame)     # e.g. filter joints, normalize, etc.
            self.load(processed)              # Fivetran loads into BigQuery table
