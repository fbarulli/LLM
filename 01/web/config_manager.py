import json
import traceback
from typing import Dict, Any
from logger_config import logger

def load_config(filepath: str) -> Dict[str, Any]:
    """
    Loads a JSON config file. 
    Crashes immediately if the file is missing or malformed.
    """
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception:
        logger.error(f"CRITICAL: Could not load config at {filepath}:\n{traceback.format_exc()}")
        raise
