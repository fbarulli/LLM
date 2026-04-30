import sys
import logging
import time
from functools import wraps
from typing import Callable, Any

# ==========================================
# 1. MAIN APPLICATION LOGGER (pipeline_output.log)
# ==========================================
logger = logging.getLogger("rag_pipeline")
logger.setLevel(logging.INFO)
logger.propagate = False 

log_format = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s')

# delay=False forces the file to open and write immediately
file_handler = logging.FileHandler('pipeline_output.log', mode='w', delay=False)
file_handler.setFormatter(log_format)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_format)

if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


# ==========================================
# 2. METRICS LOGGER (metrics.log) - AUTO FLUSH
# ==========================================
metrics_logger = logging.getLogger("metrics_pipeline")
metrics_logger.setLevel(logging.INFO)
metrics_logger.propagate = False

metrics_format = logging.Formatter('%(asctime)s [METRICS] - %(message)s')

metrics_file_handler = logging.FileHandler('metrics.log', mode='w', delay=False)
metrics_file_handler.setFormatter(metrics_format)

# --- THE FIX ---
# This custom emit ensures data skips the buffer and saves to disk immediately!
original_emit = metrics_file_handler.emit
def auto_flush_emit(record):
    original_emit(record)
    metrics_file_handler.flush()

metrics_file_handler.emit = auto_flush_emit
# ---------------

if not metrics_logger.handlers:
    metrics_logger.addHandler(metrics_file_handler)


# ==========================================
# 3. DECORATOR
# ========================== ================
def time_logger(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start: float = time.time()
        result: Any = func(*args, **kwargs)
        duration: float = time.time() - start
        
        # This writes directly to metrics.log and triggers the auto-flush
        metrics_logger.info(f"{func.__name__} executed in {duration:.4f}s")
        return result
    return wrapper


# ==========================================
# 4. DECORATED FUNCTIONS
# ==========================================
@time_logger
def retrieve_faq() -> int:
    time.sleep(0.5) # Simulating DB latency
    logger.info("Found 3 matching FAQ records.")
    return 3

@time_logger
def call_llm() -> str:
    time.sleep(1.5) # Simulating LLM API latency
    logger.info("Prompt built. Length: 1462 chars | Tokens: 325")
    logger.info("LLM Success! Total call tokens used: 413")
    return "LLM Response"


# ==========================================
# 5. EXECUTION
# ==========================================
if __name__ == "__main__":
    print("Starting pipeline...")
    
    try:
        retrieve_faq()
        call_llm()
        
        # Simulated long-running Gradio loop
        print("\nGradio UI running... Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping server via Ctrl+C...")
    finally:
        logging.shutdown()
        print("Done. Check 'metrics.log' - data is physically saved!")
