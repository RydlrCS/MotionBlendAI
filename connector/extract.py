"""
Extract module: list and read files from GCS
"""
from typing import Iterator, Dict, Any, Optional
from google.cloud import storage
from logging_util import get_logger, log_operation
from retry_util import retry_with_backoff

logger = get_logger(__name__)

@retry_with_backoff(max_attempts=5)
def list_blobs(
    bucket_name: str,
    prefix: str,
    limit: Optional[int] = None
) -> Iterator[Dict[str, Any]]:
    """
    List blobs from GCS bucket with optional limit
    
    Args:
        bucket_name: GCS bucket name (e.g., 'motionblend-mocap')
        prefix: Blob prefix to filter (e.g., 'mocap/seed/')
        limit: Maximum number of blobs to return
        
    Yields:
        Dict with file_uri, updated_at, size, name
    """
    with log_operation(logger, "list_blobs", bucket=bucket_name, 
                       prefix=prefix, limit=limit) as log:
        
        try:
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blobs = bucket.list_blobs(prefix=prefix)
            
            count = 0
            for blob in blobs:
                # Skip directories
                if blob.name.endswith('/'):
                    continue
                    
                # Only process .bvh files
                if not blob.name.lower().endswith('.bvh'):
                    continue
                
                record = {
                    "file_uri": f"gs://{bucket_name}/{blob.name}",
                    "updated_at": blob.updated.isoformat() if blob.updated else None,
                    "size": blob.size,
                    "name": blob.name.split('/')[-1]
                }
                
                log.info("blob_found", **record)
                yield record
                
                count += 1
                if limit and count >= limit:
                    log.info("limit_reached", count=count)
                    break
            
            log.info("list_complete", total_blobs=count)
            
        except Exception as e:
            log.error("list_failed", error=str(e), exc_info=True)
            raise


def infer_category(file_uri: str) -> str:
    """
    Infer motion category from GCS URI
    
    Args:
        file_uri: GCS URI like gs://bucket/mocap/seed/file.bvh
        
    Returns:
        Category: 'seed', 'build', or 'blend'
    """
    if '/seed/' in file_uri:
        return 'seed'
    elif '/build/' in file_uri:
        return 'build'
    elif '/blend/' in file_uri:
        return 'blend'
    else:
        logger.warning("unknown_category", file_uri=file_uri)
        return 'unknown'
