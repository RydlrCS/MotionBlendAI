"""
Structured logging for Fivetran connector
JSON logs with correlation IDs, timing, and contextual metadata
"""
import logging
import structlog
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Optional

# Configure structlog
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

def get_logger(name: str, correlation_id: Optional[str] = None) -> structlog.BoundLogger:
    """
    Get a structured logger with optional correlation ID
    
    Args:
        name: Logger name (usually __name__)
        correlation_id: Optional trace ID for request correlation
    
    Returns:
        Configured structlog logger
    """
    logger = structlog.get_logger(name)
    if correlation_id:
        logger = logger.bind(correlation_id=correlation_id)
    return logger

@contextmanager
def log_operation(
    logger: structlog.BoundLogger,
    operation: str,
    service: str = "connector",
    **context: Any
):
    """
    Context manager for logging operations with timing
    
    Usage:
        with log_operation(logger, "sync_files", bucket="my-bucket"):
            # do work
            pass
    """
    start_time = time.time()
    correlation_id = str(uuid.uuid4())
    
    bound_logger = logger.bind(
        service=service,
        op=operation,
        correlation_id=correlation_id,
        **context
    )
    
    bound_logger.info("operation_start")
    
    success = False
    error_code = None
    
    try:
        yield bound_logger
        success = True
    except Exception as e:
        error_code = type(e).__name__
        bound_logger.error(
            "operation_failed",
            error_code=error_code,
            error_message=str(e),
            exc_info=True
        )
        raise
    finally:
        duration_ms = int((time.time() - start_time) * 1000)
        bound_logger.info(
            "operation_complete",
            success=success,
            duration_ms=duration_ms,
            error_code=error_code
        )

class CorrelationIDFilter(logging.Filter):
    """Add correlation ID to all log records"""
    
    def __init__(self, correlation_id: Optional[str] = None):
        super().__init__()
        self.correlation_id = correlation_id or str(uuid.uuid4())
    
    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = self.correlation_id
        return True
