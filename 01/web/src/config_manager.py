# /home/admin/LLM/LLM/01/web/src/config_manager.py

import json
import traceback
from typing import Dict, Any
from src.logger_config import logger

def load_config(filepath: str) -> Dict[str, Any]:
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception:
        logger.error(f"CRITICAL: Could not load config at {filepath}:\n{traceback.format_exc()}")
        raise