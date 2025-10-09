# MotionBlend AI Fivetran Connector

A Fivetran Connector SDK integration that streams motion capture processing data, blend results, and analytics to data warehouses for advanced analytics and reporting.

## Connector Overview

This connector extracts and syncs data from a MotionBlendAI workspace to Fivetran-supported destinations, enabling analytics on:

- **Motion Capture Sequences**: Catalog of available motion data with metadata
- **Blend Processing Jobs**: Execution history and performance metrics  
- **Generated Artifacts**: Output files, manifests, and processing logs
- **User Interactions**: Analytics from the OBS-style mixer interface

## Features

- **Incremental Sync**: Only processes changed files since last sync
- **Comprehensive Metadata**: Extracts frame counts, joint counts, durations from GLB/TRC/NPY files
- **Performance Tracking**: Monitors blend job execution times and resource usage
- **Artifact Management**: Catalogs all generated files with retention policies
- **Error Handling**: Graceful handling of corrupt files and extraction errors
- **Flexible Configuration**: Configurable file size limits and sync options

## Requirements

```
fivetran-connector-sdk
numpy
pygltflib  # For GLB file processing
```

Note: The `fivetran_connector_sdk` package is pre-installed in the Fivetran environment.

## Configuration

Create a `configuration.json` file with the following parameters:

```json
{
  "workspace_path": "/path/to/MotionBlendAI-1",
  "sync_mode": "incremental",
  "include_artifacts": true,
  "max_file_size_mb": 100,
  "sync_ui_analytics": true
}
```

### Configuration Parameters

- **workspace_path** (required): Absolute path to MotionBlendAI workspace root
- **sync_mode** (optional): `"incremental"` or `"full"` sync mode (default: incremental)
- **include_artifacts** (optional): Whether to sync processing artifacts (default: true)
- **max_file_size_mb** (optional): Maximum file size to process in MB (default: 100)
- **sync_ui_analytics** (optional): Include UI interaction data (default: true)

## Authentication

This connector accesses local filesystem data and does not require external authentication. Ensure the Fivetran service has read access to the configured workspace path.

## Data Sources

The connector scans the following directories in the workspace:

- `build/build_motions/` - Source motion capture files
- `build/blend_snn/` - SNN blend results  
- `project/seed_motions/` - **Seed motion library (primary source)**
- `seed_motions/` - Root-level seed motions (if present)
- `build/demo_artifacts/` - Generated artifacts and manifests

### Supported File Formats

- **GLB**: 3D motion files with skeletal animation
- **TRC**: Motion capture marker data
- **FBX**: Autodesk motion files (requires fbx2json tool)
- **NPY**: NumPy arrays with motion data
- **JSON**: Manifests and metadata files
- **LOG**: Processing and error logs

## Tables Created

### motion_sequences
Catalog of available motion capture data with extracted metadata:

```sql
CREATE TABLE motion_sequences (
  sequence_id VARCHAR PRIMARY KEY,
  name VARCHAR,
  file_path VARCHAR,
  file_format VARCHAR,
  file_size_bytes BIGINT,
  frame_count INTEGER,
  joint_count INTEGER,
  duration_seconds FLOAT,
  fps FLOAT,
  created_at TIMESTAMP,
  last_modified TIMESTAMP,
  discovered_at TIMESTAMP,
  metadata JSON,
  is_active BOOLEAN
);
```

### blend_jobs
History of motion blending operations with performance metrics:

```sql
CREATE TABLE blend_jobs (
  job_id VARCHAR PRIMARY KEY,
  sequence_a_id VARCHAR,
  sequence_b_id VARCHAR,
  blend_weight FLOAT,
  blend_method VARCHAR,
  status VARCHAR,
  started_at TIMESTAMP,
  completed_at TIMESTAMP,
  duration_ms BIGINT,
  output_file_path VARCHAR,
  output_file_size_bytes BIGINT,
  output_frame_count INTEGER,
  error_message VARCHAR,
  initiated_by VARCHAR,
  processing_config JSON,
  performance_metrics JSON
);
```

### processing_artifacts
Generated files and their characteristics:

```sql
CREATE TABLE processing_artifacts (
  artifact_id VARCHAR PRIMARY KEY,
  job_id VARCHAR,
  artifact_type VARCHAR,
  file_name VARCHAR,
  file_path VARCHAR,
  file_size_bytes BIGINT,
  mime_type VARCHAR,
  created_at TIMESTAMP,
  is_temporary BOOLEAN,
  retention_days INTEGER,
  download_count INTEGER,
  last_accessed TIMESTAMP,
  metadata JSON,
  checksum_sha256 VARCHAR
);
```

### ui_interactions
User behavior analytics from the mixer interface:

```sql
CREATE TABLE ui_interactions (
  interaction_id VARCHAR PRIMARY KEY,
  session_id VARCHAR,
  interaction_type VARCHAR,
  sequence_id VARCHAR,
  timestamp TIMESTAMP,
  parameters JSON,
  duration_ms BIGINT,
  resulted_in_blend BOOLEAN,
  user_agent VARCHAR,
  viewport_size VARCHAR,
  performance_timing JSON
);
```

## Data Handling

1. **Motion Sequence Discovery**: Scans workspace directories for motion files
2. **Metadata Extraction**: Uses specialized extractors for each file format
3. **Incremental Processing**: Tracks file modification times to avoid reprocessing
4. **Blend Job Reconstruction**: Infers job history from output files and naming patterns
5. **Artifact Cataloging**: Comprehensive tracking of all generated files
6. **State Management**: Maintains sync checkpoints for reliable incremental updates

## Error Handling

- **File Access Errors**: Logs warnings and continues processing other files
- **Corrupt Files**: Graceful handling with error metadata
- **Missing Dependencies**: Falls back to basic file information when specialized tools unavailable
- **Configuration Validation**: Clear error messages for invalid configuration
- **State Recovery**: Robust checkpoint system for interrupted syncs

## Local Testing

```bash
# Install dependencies
pip install fivetran-connector-sdk numpy pygltflib

# Test the connector locally
cd project/fivetran_connector
python connector.py

# Debug with Fivetran CLI
fivetran debug --configuration configuration.json

# Reset state for fresh sync
fivetran reset
```

## Deployment

```bash
# Deploy to Fivetran
fivetran deploy \
  --api-key <FIVETRAN_API_KEY> \
  --destination <DESTINATION_ID> \
  --connection <CONNECTION_ID> \
  --configuration configuration.json
```

## Analytics Use Cases

With this data in your warehouse, you can analyze:

- **Motion Library Growth**: Track sequence discovery over time
- **Blend Operation Patterns**: Most popular sequence combinations
- **Processing Performance**: Blend job duration and success rates
- **User Engagement**: Interaction patterns in the mixer interface
- **Data Quality**: File corruption rates and extraction success
- **Resource Usage**: Storage growth and artifact lifecycle

## Example Queries

```sql
-- Top 10 most blended sequences
SELECT 
  sequence_a_id,
  COUNT(*) as blend_count,
  AVG(duration_ms) as avg_duration_ms
FROM blend_jobs 
WHERE status = 'completed'
GROUP BY sequence_a_id
ORDER BY blend_count DESC
LIMIT 10;

-- Daily processing volume
SELECT 
  DATE(completed_at) as date,
  COUNT(*) as jobs_completed,
  SUM(output_file_size_bytes) as total_output_bytes
FROM blend_jobs
WHERE status = 'completed'
GROUP BY DATE(completed_at)
ORDER BY date DESC;

-- Motion sequence catalog with metadata
SELECT 
  name,
  file_format,
  frame_count,
  joint_count,
  duration_seconds,
  file_size_bytes / 1024 / 1024 as size_mb
FROM motion_sequences
WHERE is_active = true
ORDER BY frame_count DESC;
```

## Troubleshooting

### Common Issues

1. **Workspace Path Not Found**
   - Verify the workspace_path in configuration.json
   - Ensure Fivetran has read access to the directory

2. **No Motion Files Detected**
   - Check that motion files exist in expected directories
   - Verify file permissions are readable

3. **Metadata Extraction Failures**
   - Install required dependencies (numpy, pygltflib)
   - Check file formats are supported

4. **Large File Processing**
   - Adjust max_file_size_mb in configuration
   - Consider excluding large files from sync

### Debug Logging

Enable detailed logging by running the connector locally:

```python
from fivetran_connector_sdk import Logging as log
log.set_level('DEBUG')
```

## Additional Considerations

This connector is provided as an example implementation for integrating MotionBlendAI data with Fivetran. While tested, users should validate the connector meets their specific requirements before production deployment.

For questions or support, refer to the [Fivetran Connector SDK documentation](https://fivetran.com/docs/connectors/connector-sdk) or contact Fivetran support.