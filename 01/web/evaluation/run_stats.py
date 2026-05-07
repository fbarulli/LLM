# /home/admin/LLM/LLM/01/web/src/run_stats.py

import json
import os
from typing import List, Dict
from src.core import generate_document_id

def get_eval_set(filename: str = "documents.json", n_per_course: int = 10) -> List[Dict]:
    # If a full path isn't provided, build it relative to the web root (parent of src)
    if not os.path.isabs(filename):
        # Get the directory where this file is (src), then go up one level to web root
        src_dir = os.path.dirname(os.path.abspath(__file__))
        web_root = os.path.dirname(src_dir)
        filename = os.path.join(web_root, filename)
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Could not find ground truth file at: {filename}")

    with open(filename, 'r') as f:
        raw_data = json.load(f)
        
    eval_set = []
    for course_entry in raw_data:
        c_name = course_entry["course"]
        for i, doc in enumerate(course_entry["documents"][:n_per_course]):
            # Clean the question before generating ID
            clean_question = doc["question"]
            if " - " in clean_question:
                clean_question = clean_question.split(" - ", 1)[1].strip()
            
            clean_doc = {
                "text": doc["text"],
                "question": clean_question,
                "course": c_name
            }
            eval_set.append({
                "query": clean_question,
                "course": c_name,
                "expected_id": generate_document_id(clean_doc)
            })
    return eval_set