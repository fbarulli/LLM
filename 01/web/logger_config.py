import sys
import logging
import time
from functools import wraps
from typing import Callable, Any, Optional, Tuple

# --- logger_config.py ---


logger = logging.getLogger("rag_pipeline")
logger.setLevel(logging.INFO)
logger.propagate = False 

log_format = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s')

file_handler = logging.FileHandler('pipeline_output.log', mode='w', delay=False)
file_handler.setFormatter(log_format)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_format)

if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

metrics_logger = logging.getLogger("metrics_pipeline")
metrics_logger.setLevel(logging.INFO)
metrics_logger.propagate = False

metrics_format = logging.Formatter('%(asctime)s [METRICS] - %(message)s')


metrics_file_handler = logging.FileHandler('metrics.log', mode='w', delay=False)
metrics_file_handler.setFormatter(metrics_format)

if not metrics_logger.handlers:
    metrics_logger.addHandler(metrics_file_handler)

def time_logger(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger.info(f"START: Calling function '{func.__name__}'")
        start: float = time.time()
        status = "SUCCESS"
        
        try:
            result: Any = func(*args, **kwargs)
            return result
        except Exception as e:
            status = "FAILED"
            logger.error(f"ERROR: Function '{func.__name__}' failed with: {str(e)}")
            raise
        finally:
            # Ensures metrics are captured even on timeouts/errors
            duration: float = time.time() - start
            metrics_message = f"{func.__name__} [{status}] executed in {duration:.4f}s"
            metrics_logger.info(metrics_message)
            
            # Force flush and close to ensure writing to disk
            metrics_file_handler.flush()
            for handler in logger.handlers:
                handler.flush()
                
            logger.info(f"END: Function '{func.__name__}' finished in {duration:.4f}s")
            
    return wrapper