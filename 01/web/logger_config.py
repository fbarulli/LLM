import sys
import logging
import time
from functools import wraps
from typing import Callable, Any

logger = logging.getLogger("rag_pipeline")
logger.setLevel(logging.INFO)
logger.propagate = False 

log_format = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s')

file_handler = logging.FileHandler('pipeline_output.log', mode='w')
file_handler.setFormatter(log_format)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_format)

if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def time_logger(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to time function execution and log it.
    (i) func / (o) wrapper
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start: float = time.time()
        result: Any = func(*args, **kwargs)
        duration: float = time.time() - start
        logger.info(f"{func.__name__} executed in {duration:.4f}s")
        return result
    return wrapper
