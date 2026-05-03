import json
from typing import List, Dict
from stats import StatsCollector
from core import generate_document_id

def get_eval_set(filepath: str, n_per_course: int = 10) -> List[Dict]:
    with open(filepath, 'r') as f:
        raw_data = json.load(f)
    eval_set = []
    for course_entry in raw_data:
        c_name = course_entry["course"]
        for i, doc in enumerate(course_entry["documents"]):
            if i >= n_per_course: break
            clean_doc = {"text": doc["text"], "question": doc["question"], "course": c_name}
            eval_set.append({
                "query": doc["question"], "course": c_name, "expected_text": doc["text"],
                "expected_id": generate_document_id(clean_doc)
            })
    return eval_set

if __name__ == "__main__":
    # Point to the specific config for this run
    collector = StatsCollector("experiments/configs/baseline_bm25.json")
    eval_data = get_eval_set("documents.json")
    collector.run_benchmark(eval_data, "baseline_bm25")
