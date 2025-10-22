"""
Fivetran connector orchestrator
Syncs files from GCS → Transform → Load to BigQuery
"""
import argparse
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from connector.logging_util import get_logger, log_operation
from connector.retry_util import retry_with_backoff
from connector import discover, extract, transform, load, state

logger = get_logger(__name__)

def sync_category(
    bucket: str,
    prefix: str,
    category: str,
    dataset: str,
    limit: int = None,
    project: str = None
) -> int:
    """
    Sync one category of motion files
    
    Args:
        bucket: GCS bucket name
        prefix: Blob prefix
        category: Motion category (seed/build/blend)
        dataset: BigQuery dataset
        limit: Max files to process
        project: GCP project ID
        
    Returns:
        Number of records processed
    """
    with log_operation(logger, "sync_category", bucket=bucket, prefix=prefix, 
                       category=category, limit=limit) as log:
        
        # Get current cursor
        cursor = state.get_cursor(category)
        log.info("cursor_loaded", cursor=cursor)
        
        # Extract: list blobs from GCS
        blobs = list(extract.list_blobs(bucket, prefix, limit))
        log.info("extraction_complete", blob_count=len(blobs))
        
        if not blobs:
            log.info("no_blobs_found", message="Nothing to sync")
            return 0
        
        # Transform: normalize records
        records = []
        for blob in blobs:
            try:
                normalized = transform.normalize_record(blob, category)
                records.append(normalized)
            except Exception as e:
                log.error("transform_failed", blob=blob, error=str(e))
                continue
        
        log.info("transformation_complete", record_count=len(records))
        
        if not records:
            log.info("no_records_to_load")
            return 0
        
        # Load: insert into BigQuery
        table_name = load.get_table_name(category)
        
        # Create table if it doesn't exist
        schema = load.SCHEMA_MAP[category]
        load.create_table_if_not_exists(dataset, table_name, schema, project)
        
        # Insert rows
        load.upsert_rows(dataset, table_name, records, project)
        
        # Update state
        last_blob = blobs[-1]
        new_cursor = last_blob.get("updated_at") or last_blob.get("name")
        state.update_cursor(category, new_cursor)
        state.record_sync(category, len(records))
        
        log.info("sync_complete", records_synced=len(records))
        return len(records)


def main():
    parser = argparse.ArgumentParser(
        description="Fivetran connector: GCS → BigQuery"
    )
    parser.add_argument("--bucket", default="motionblend-mocap",
                       help="GCS bucket name")
    parser.add_argument("--prefix", default="mocap/seed/",
                       help="GCS prefix to sync")
    parser.add_argument("--category", 
                       choices=["seed", "build", "blend"],
                       help="Motion category (inferred from prefix if not set)")
    parser.add_argument("--dataset", default="RAW_DEV",
                       help="BigQuery dataset")
    parser.add_argument("--project", 
                       default=os.getenv("GCP_PROJECT"),
                       help="GCP project ID")
    parser.add_argument("--limit", type=int, default=2,
                       help="Max files to process (for testing)")
    parser.add_argument("--discover-only", action="store_true",
                       help="Only run discovery (print schema)")
    
    args = parser.parse_args()
    
    logger.info("connector_start", bucket=args.bucket, prefix=args.prefix,
               dataset=args.dataset, project=args.project)
    
    try:
        # Discovery mode
        if args.discover_only:
            catalog = discover.discover()
            print("\n📋 Schema Catalog:")
            for table, columns in catalog.items():
                print(f"\n  {table}:")
                for col in columns:
                    print(f"    - {col}")
            return
        
        # Infer category from prefix if not provided
        category = args.category
        if not category:
            category = extract.infer_category(f"gs://{args.bucket}/{args.prefix}")
            logger.info("category_inferred", category=category)
        
        if category == "unknown":
            raise ValueError(f"Cannot infer category from prefix: {args.prefix}")
        
        # Run sync
        records_synced = sync_category(
            bucket=args.bucket,
            prefix=args.prefix,
            category=category,
            dataset=args.dataset,
            limit=args.limit,
            project=args.project
        )
        
        logger.info("connector_success", records_synced=records_synced)
        print(f"\n✅ Successfully synced {records_synced} records")
        print(f"   Category: {category}")
        print(f"   Dataset: {args.dataset}")
        print(f"   Table: {load.get_table_name(category)}")
        
    except Exception as e:
        logger.error("connector_failed", error=str(e), exc_info=True)
        print(f"\n❌ Connector failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
