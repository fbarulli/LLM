import json
import os
from typing import List, Dict
from core import generate_document_id

def get_eval_set(filename: str = "documents.json", n_per_course: int = 10) -> List[Dict]:
    # If a full path isn't provided, build it relative to this file's location
    if not os.path.isabs(filename):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(base_dir, filename)
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Could not find ground truth file at: {filename}")

    with open(filename, 'r') as f:
        raw_data = json.load(f)
        
    eval_set = []
    for course_entry in raw_data:
        c_name = course_entry["course"]
        # Use a slice to get questions from EACH course
        for i, doc in enumerate(course_entry["documents"][:n_per_course]):
            clean_doc = {"text": doc["text"], "question": doc["question"], "course": c_name}
            eval_set.append({
                "query": doc["question"], 
                "course": c_name, 
                "expected_id": generate_document_id(clean_doc)
            })
    return eval_set
