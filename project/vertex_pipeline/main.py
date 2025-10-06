# Entry point for GCP Cloud Function: trigger_vertex_pipeline
# This file must be named main.py for Cloud Functions deployment

from flask import Request
from trigger_pipeline_function import http_trigger

def main(request: Request):
	"""GCP Cloud Function entry point."""
	return http_trigger(request)
