"""
Retry logic with exponential backoff, jitter, and circuit breaker
"""
import time
import random
from typing import Callable, Optional, Type, Tuple
from functools import wraps
import structlog

logger = structlog.get_logger(__name__)

class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open"""
    pass

class CircuitBreaker:
    """
    Circuit breaker pattern implementation
    Trips open if failure rate exceeds threshold within window
    """
    
    def __init__(
        self,
        failure_threshold: float = 0.5,
        window_seconds: int = 300,
        cooldown_seconds: int = 120
    ):
        self.failure_threshold = failure_threshold
        self.window_seconds = window_seconds
        self.cooldown_seconds = cooldown_seconds
        
        self.failures = []
        self.successes = []
        self.opened_at: Optional[float] = None
        self.state = "closed"  # closed, open, half_open
    
    def _clean_old_records(self):
        """Remove records outside the time window"""
        cutoff = time.time() - self.window_seconds
        self.failures = [t for t in self.failures if t > cutoff]
        self.successes = [t for t in self.successes if t > cutoff]
    
    def record_success(self):
        """Record a successful operation"""
        self._clean_old_records()
        self.successes.append(time.time())
        
        if self.state == "half_open":
            logger.info("circuit_breaker_closed", state="half_open->closed")
            self.state = "closed"
            self.opened_at = None
    
    def record_failure(self):
        """Record a failed operation"""
        self._clean_old_records()
        self.failures.append(time.time())
        
        total = len(self.failures) + len(self.successes)
        if total >= 5:  # Min sample size
            failure_rate = len(self.failures) / total
            if failure_rate > self.failure_threshold:
                self.state = "open"
                self.opened_at = time.time()
                logger.warning(
                    "circuit_breaker_opened",
                    failure_rate=failure_rate,
                    threshold=self.failure_threshold
                )
    
    def allow_request(self) -> bool:
        """Check if request should be allowed"""
        if self.state == "closed":
            return True
        
        if self.state == "open":
            if time.time() - self.opened_at > self.cooldown_seconds:
                logger.info("circuit_breaker_half_open", state="open->half_open")
                self.state = "half_open"
                return True
            return False
        
        # half_open state
        return True

def retry_with_backoff(
    max_attempts: int = 7,
    base_delay: float = 0.5,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    circuit_breaker: Optional[CircuitBreaker] = None
):
    """
    Decorator for retrying with exponential backoff and jitter
    
    Args:
        max_attempts: Maximum retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap
        backoff_factor: Exponential multiplier
        jitter: Add randomness to prevent thundering herd
        exceptions: Tuple of exceptions to retry on
        circuit_breaker: Optional circuit breaker instance
    
    Example:
        @retry_with_backoff(max_attempts=5)
        def fetch_data():
            response = requests.get("https://api.example.com")
            response.raise_for_status()
            return response.json()
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            delay = base_delay
            
            while attempt < max_attempts:
                # Check circuit breaker
                if circuit_breaker and not circuit_breaker.allow_request():
                    logger.error("circuit_breaker_rejection", func=func.__name__)
                    raise CircuitBreakerOpen(f"Circuit breaker open for {func.__name__}")
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Record success
                    if circuit_breaker:
                        circuit_breaker.record_success()
                    
                    if attempt > 0:
                        logger.info(
                            "retry_success",
                            func=func.__name__,
                            attempt=attempt,
                            total_attempts=max_attempts
                        )
                    
                    return result
                
                except exceptions as e:
                    attempt += 1
                    
                    # Record failure
                    if circuit_breaker:
                        circuit_breaker.record_failure()
                    
                    if attempt >= max_attempts:
                        logger.error(
                            "retry_exhausted",
                            func=func.__name__,
                            attempts=attempt,
                            error=str(e)
                        )
                        raise
                    
                    # Calculate delay with exponential backoff
                    current_delay = min(delay, max_delay)
                    
                    # Add jitter to prevent thundering herd
                    if jitter:
                        current_delay = current_delay * (0.5 + random.random())
                    
                    logger.warning(
                        "retry_attempt",
                        func=func.__name__,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        delay_seconds=current_delay,
                        error=str(e)
                    )
                    
                    time.sleep(current_delay)
                    delay *= backoff_factor
            
            raise RuntimeError(f"Max retries exceeded for {func.__name__}")
        
        return wrapper
    return decorator
