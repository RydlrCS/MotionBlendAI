"""
Batch exporter: BigQuery MARTS → Elasticsearch
Designed to run as a Cloud Run job or locally
"""
import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Iterator

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from google.cloud import bigquery
from elasticsearch import Elasticsearch, helpers
from connector.logging_util import get_logger, log_operation
from connector.retry_util import retry_with_backoff

logger = get_logger(__name__)

# Elasticsearch index template
INDEX_TEMPLATE = {
    "mappings": {
        "properties": {
            "blend_id": {"type": "keyword"},
            "left_motion_id": {"type": "keyword"},
            "right_motion_id": {"type": "keyword"},
            "blend_ratio": {"type": "float"},
            "transition_start_frame": {"type": "integer"},
            "transition_end_frame": {"type": "integer"},
            "method": {"type": "keyword"},
            # Quality metrics
            "fid": {"type": "float"},
            "coverage": {"type": "float"},
            "gdiv": {"type": "float"},
            "ldiv": {"type": "float"},
            "inter_div": {"type": "float"},
            "intra_div": {"type": "float"},
            # Velocity/acceleration
            "l2_velocity_mean": {"type": "float"},
            "l2_acceleration_mean": {"type": "float"},
            "transition_smoothness": {"type": "float"},
            # Quality derived
            "quality_score": {"type": "float"},
            "quality_category": {"type": "keyword"},
            # Timestamps
            "created_at": {"type": "date"},
            "computed_at": {"type": "date"}
        }
    },
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "refresh_interval": "30s"
    }
}


@retry_with_backoff(max_attempts=3)
def ensure_index(es: Elasticsearch, index: str) -> None:
    """
    Create index with template if it doesn't exist
    
    Args:
        es: Elasticsearch client
        index: Index name
    """
    with log_operation(logger, "ensure_index", index=index) as log:
        if es.indices.exists(index=index):
            log.info("index_exists")
            return
        
        es.indices.create(index=index, body=INDEX_TEMPLATE)
        log.info("index_created")


def query_bigquery(
    project: str,
    dataset: str,
    table: str,
    batch_size: int = 100
) -> Iterator[List[Dict[str, Any]]]:
    """
    Query BigQuery table in batches
    
    Args:
        project: GCP project ID
        dataset: BigQuery dataset
        table: Table name
        batch_size: Records per batch
        
    Yields:
        Batches of records as dictionaries
    """
    with log_operation(logger, "query_bigquery", project=project, 
                       dataset=dataset, table=table) as log:
        
        client = bigquery.Client(project=project)
        query = f"""
            SELECT
                blend_id,
                left_motion_id,
                right_motion_id,
                blend_ratio,
                transition_start_frame,
                transition_end_frame,
                method,
                fid,
                coverage,
                gdiv,
                ldiv,
                inter_div,
                intra_div,
                l2_velocity_mean,
                l2_acceleration_mean,
                transition_smoothness,
                quality_score,
                quality_category,
                TIMESTAMP_SECONDS(created_at) as created_at,
                computed_at
            FROM `{project}.{dataset}.{table}`
        """
        
        log.info("executing_query", query=query[:200])
        result = client.query(query).result()
        
        batch = []
        total = 0
        
        for row in result:
            batch.append(dict(row))
            
            if len(batch) >= batch_size:
                total += len(batch)
                log.info("batch_ready", batch_size=len(batch), total=total)
                yield batch
                batch = []
        
        # Yield remaining records
        if batch:
            total += len(batch)
            log.info("final_batch", batch_size=len(batch), total=total)
            yield batch
        
        log.info("query_complete", total_records=total)


@retry_with_backoff(max_attempts=3)
def bulk_index(
    es: Elasticsearch,
    index: str,
    records: List[Dict[str, Any]]
) -> int:
    """
    Bulk index records to Elasticsearch
    
    Args:
        es: Elasticsearch client
        index: Target index
        records: Records to index
        
    Returns:
        Number of records indexed
    """
    with log_operation(logger, "bulk_index", index=index, 
                       record_count=len(records)) as log:
        
        if not records:
            log.info("no_records_to_index")
            return 0
        
        # Prepare actions for bulk API
        actions = []
        for record in records:
            action = {
                "_index": index,
                "_id": record["blend_id"],
                "_source": record
            }
            actions.append(action)
        
        # Execute bulk operation
        success, failed = helpers.bulk(
            es, 
            actions,
            raise_on_error=False,
            raise_on_exception=False
        )
        
        if failed:
            log.warning("bulk_partial_failure", success=success, failed=len(failed))
            # Log first few failures for debugging
            for i, failure in enumerate(failed[:3]):
                log.error("index_failure", error=failure)
        else:
            log.info("bulk_success", indexed=success)
        
        return success


def export_table(
    bq_project: str,
    bq_dataset: str,
    bq_table: str,
    es_url: str,
    es_index: str,
    es_api_key: str = None,
    batch_size: int = 100
) -> int:
    """
    Export BigQuery table to Elasticsearch
    
    Args:
        bq_project: BigQuery project ID
        bq_dataset: BigQuery dataset
        bq_table: Table name
        es_url: Elasticsearch URL
        es_index: Target index name
        es_api_key: Elasticsearch API key (optional)
        batch_size: Records per batch
        
    Returns:
        Total records exported
    """
    with log_operation(logger, "export_table", 
                       bq_table=f"{bq_project}.{bq_dataset}.{bq_table}",
                       es_index=es_index) as log:
        
        # Initialize Elasticsearch client
        es_config = {"hosts": [es_url]}
        if es_api_key:
            es_config["api_key"] = es_api_key
        
        es = Elasticsearch(**es_config)
        log.info("es_connected", url=es_url, cluster_name=es.info()["cluster_name"])
        
        # Ensure index exists
        ensure_index(es, es_index)
        
        # Export in batches
        total_exported = 0
        for batch in query_bigquery(bq_project, bq_dataset, bq_table, batch_size):
            indexed = bulk_index(es, es_index, batch)
            total_exported += indexed
            log.info("batch_exported", total=total_exported)
        
        log.info("export_complete", total_records=total_exported)
        return total_exported


def main():
    parser = argparse.ArgumentParser(
        description="Export BigQuery MARTS → Elasticsearch"
    )
    parser.add_argument("--bq-project", 
                       default=os.getenv("GCP_PROJECT"),
                       required=True,
                       help="BigQuery project ID")
    parser.add_argument("--bq-dataset", default="RAW_DEV",
                       help="BigQuery dataset")
    parser.add_argument("--bq-table", default="mart_blend_snn_metrics",
                       help="BigQuery table name")
    parser.add_argument("--es-url",
                       default=os.getenv("ELASTICSEARCH_URL"),
                       required=True,
                       help="Elasticsearch URL")
    parser.add_argument("--es-index", default="mb_blends_v1",
                       help="Elasticsearch index name")
    parser.add_argument("--es-api-key",
                       default=os.getenv("ELASTICSEARCH_API_KEY"),
                       help="Elasticsearch API key")
    parser.add_argument("--batch-size", type=int, default=100,
                       help="Records per batch")
    
    args = parser.parse_args()
    
    logger.info("exporter_start", 
               bq_project=args.bq_project,
               bq_dataset=args.bq_dataset,
               bq_table=args.bq_table,
               es_url=args.es_url,
               es_index=args.es_index)
    
    try:
        total = export_table(
            bq_project=args.bq_project,
            bq_dataset=args.bq_dataset,
            bq_table=args.bq_table,
            es_url=args.es_url,
            es_index=args.es_index,
            es_api_key=args.es_api_key,
            batch_size=args.batch_size
        )
        
        logger.info("exporter_success", total_exported=total)
        print(f"\n✅ Successfully exported {total} records")
        print(f"   From: {args.bq_project}.{args.bq_dataset}.{args.bq_table}")
        print(f"   To: {args.es_index}")
        
    except Exception as e:
        logger.error("exporter_failed", error=str(e), exc_info=True)
        print(f"\n❌ Export failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
