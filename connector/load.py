"""
Load module: insert records into BigQuery
"""
from typing import List, Dict, Any
from google.cloud import bigquery
from logging_util import get_logger, log_operation
from retry_util import retry_with_backoff

logger = get_logger(__name__)

# Table name mappings
TABLE_MAP = {
    'seed': 'seed_motions',
    'build': 'build_motions',
    'blend': 'blend_snn'
}

@retry_with_backoff(max_attempts=5)
def upsert_rows(
    dataset: str,
    table: str,
    rows: List[Dict[str, Any]],
    project: str = None
) -> None:
    """
    Insert rows into BigQuery table
    
    Args:
        dataset: BigQuery dataset name (e.g., 'RAW_DEV')
        table: Table name (e.g., 'seed_motions')
        rows: List of row dictionaries
        project: GCP project ID (defaults to client default)
    """
    with log_operation(logger, "upsert_rows", dataset=dataset, 
                       table=table, row_count=len(rows)) as log:
        
        if not rows:
            log.info("no_rows_to_insert")
            return
        
        try:
            client = bigquery.Client(project=project) if project else bigquery.Client()
            table_ref = f"{client.project}.{dataset}.{table}"
            
            log.info("inserting_rows", table_ref=table_ref, count=len(rows))
            
            errors = client.insert_rows_json(table_ref, rows)
            
            if errors:
                log.error("insert_errors", errors=errors)
                raise RuntimeError(f"BigQuery insert errors: {errors}")
            
            log.info("insert_complete", rows_inserted=len(rows))
            
        except Exception as e:
            log.error("insert_failed", error=str(e), exc_info=True)
            raise


def get_table_name(category: str) -> str:
    """
    Get BigQuery table name for category
    
    Args:
        category: Motion category ('seed', 'build', 'blend')
        
    Returns:
        Table name
    """
    table = TABLE_MAP.get(category)
    if not table:
        raise ValueError(f"Unknown category: {category}. Valid: {list(TABLE_MAP.keys())}")
    return table


def create_table_if_not_exists(
    dataset: str,
    table: str,
    schema: List[bigquery.SchemaField],
    project: str = None
) -> None:
    """
    Create BigQuery table if it doesn't exist
    
    Args:
        dataset: Dataset name
        table: Table name
        schema: List of SchemaField objects
        project: GCP project ID
    """
    with log_operation(logger, "create_table", dataset=dataset, table=table) as log:
        try:
            client = bigquery.Client(project=project) if project else bigquery.Client()
            table_ref = f"{client.project}.{dataset}.{table}"
            
            # Check if table exists
            try:
                client.get_table(table_ref)
                log.info("table_exists", table_ref=table_ref)
                return
            except Exception:
                pass
            
            # Create table
            table_obj = bigquery.Table(table_ref, schema=schema)
            
            # Add partitioning for performance
            table_obj.time_partitioning = bigquery.TimePartitioning(
                field="created_at",
                type_=bigquery.TimePartitioningType.DAY
            )
            
            client.create_table(table_obj)
            log.info("table_created", table_ref=table_ref)
            
        except Exception as e:
            log.error("create_table_failed", error=str(e), exc_info=True)
            raise


# Schema definitions
SEED_SCHEMA = [
    bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("file_uri", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("skeleton_id", "STRING"),
    bigquery.SchemaField("frames", "INTEGER"),
    bigquery.SchemaField("fps", "INTEGER"),
    bigquery.SchemaField("joints_count", "INTEGER"),
    bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("updated_at", "TIMESTAMP"),
]

BUILD_SCHEMA = [
    bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("file_uri", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("skeleton_id", "STRING"),
    bigquery.SchemaField("frames", "INTEGER"),
    bigquery.SchemaField("fps", "INTEGER"),
    bigquery.SchemaField("joints_count", "INTEGER"),
    bigquery.SchemaField("build_method", "STRING"),
    bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("updated_at", "TIMESTAMP"),
]

BLEND_SCHEMA = [
    bigquery.SchemaField("id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("left_motion_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("right_motion_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("blend_ratio", "FLOAT"),
    bigquery.SchemaField("transition_start_frame", "INTEGER"),
    bigquery.SchemaField("transition_end_frame", "INTEGER"),
    bigquery.SchemaField("method", "STRING"),
    bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("updated_at", "TIMESTAMP"),
]

SCHEMA_MAP = {
    'seed': SEED_SCHEMA,
    'build': BUILD_SCHEMA,
    'blend': BLEND_SCHEMA
}
