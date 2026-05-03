import json
import time
import traceback
from typing import List, Dict, Any
from search import CourseRAGManager
from config_manager import load_config
from core import generate_document_id
from langfuse.decorators import observe

class StatsCollector:
    def __init__(self, config_path: str):
        self.settings = load_config(config_path)
        self.manager = CourseRAGManager(self.settings)
        self.manager.connect_elasticsearch()

    def _get_edit_distance(self, s1: str, s2: str) -> int:
        return abs(len(s1) - len(s2)) # Simple fallback for diagnostic

    @observe()
    def run_benchmark(self, eval_set: List[Dict], experiment_name: str) -> str:
        results = []
        k_values = [1, 3, 5, 10]

        for k in k_values:
            for item in eval_set:
                start = time.time()
                hits = self.manager.search_faq(item["query"], override_size=k)
                latency = (time.time() - start) * 1000
                
                top_hit = hits[0] if hits else None
                found_id = top_hit["_id"] if top_hit else "NONE"
                found_text = top_hit["_source"]["text"] if top_hit else ""
                
                results.append({
                    "k": k,
                    "query": item["query"],
                    "query_len": len(item["query"]),
                    "expected_course": item["course"],
                    "found_course": top_hit["_source"]["course"] if top_hit else "NONE",
                    "expected_id": item["expected_id"],
                    "found_id": found_id,
                    "success": item["expected_id"] in [h["_id"] for h in hits],
                    "score": top_hit["_score"] if top_hit else 0.0,
                    "latency_ms": round(latency, 2),
                    "edit_distance": self._get_edit_distance(item["query"], found_text),
                    "tokens_est": len(found_text) // 4
                })

        output = {
            "metadata": {"name": experiment_name, "settings": self.settings, "timestamp": time.time()},
            "results": results
        }
        
        filename = f"experiments/results/{experiment_name}.json"
        with open(filename, "w") as f:
            json.dump(output, f, indent=4)
        return filename
