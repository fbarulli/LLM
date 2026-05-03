import json
import time
import os
import traceback
from typing import List, Dict, Any
from search import CourseRAGManager
from config_manager import load_config
from core import generate_document_id
from langfuse.decorators import observe

class StatsCollector:
    """
    Handles retrieval benchmarking, atomic data collection, 
    and path-safe result storage.
    """
    
    def __init__(self, config_path: str):
        # Fail-fast: config_manager will crash if path is wrong
        self.settings = load_config(config_path)
        self.manager = CourseRAGManager(self.settings)
        self.manager.connect_elasticsearch()

    def _get_edit_distance(self, s1: str, s2: str) -> int:
        """Calculates literal gap between query and result."""
        try:
            from Levenshtein import distance
            return distance(s1, s2)
        except ImportError:
            return abs(len(s1) - len(s2))

    @observe()
    def run_benchmark(self, eval_set: List[Dict], experiment_name: str) -> str:
        """
        Executes a K-sweep across the eval set and saves results 
        to an absolute path in experiments/results/.
        """
        results = []
        k_values = [1, 3, 5, 10]

        print(f"--- 🧪 Running Experiment: {experiment_name} ---")

        for k in k_values:
            for item in eval_set:
                start_time = time.time()
                
                # 1. Determine Context (Global vs Filtered)
                # If 'global' is in the name, we pass None to disable the ES filter
                context = None
                if "global" not in experiment_name.lower():
                    context = item["course"]

                # 2. Execute Search
                hits = self.manager.search_faq(
                    query=item["query"], 
                    override_size=k, 
                    course_context=context
                )
                
                latency = (time.time() - start_time) * 1000
                
                # 3. Process Top Hit
                top_hit = hits[0] if hits else None
                found_id = top_hit["_id"] if top_hit else "NONE"
                found_text = top_hit["_source"]["text"] if top_hit else ""
                found_course = top_hit["_source"]["course"] if top_hit else "NONE"
                
                # 4. Atomic Record Creation
                results.append({
                    "k": k,
                    "query": item["query"],
                    "query_len": len(item["query"]),
                    "expected_course": item["course"],
                    "found_course": found_course,
                    "expected_id": item["expected_id"],
                    "found_id": found_id,
                    "success": item["expected_id"] in [h["_id"] for h in hits],
                    "score": top_hit["_score"] if top_hit else 0.0,
                    "latency_ms": round(latency, 2),
                    "edit_distance": self._get_edit_distance(item["query"], found_text),
                    "tokens_est": len(found_text) // 4
                })

        # --- ABSOLUTE PATH LOGIC ---
        # Ensure we always land in /web/experiments/results/
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(current_file_dir, "experiments", "results")
        os.makedirs(results_dir, exist_ok=True)
        
        filename = os.path.join(results_dir, f"{experiment_name}.json")
        
        output_payload = {
            "metadata": {
                "name": experiment_name, 
                "settings": self.settings, 
                "timestamp": time.time()
            },
            "results": results
        }
        
        # Write to disk
        try:
            with open(filename, "w") as f:
                json.dump(output_payload, f, indent=4)
            print(f"✅ SUCCESS: Saved results to {filename}")
        except Exception:
            print(f"❌ CRITICAL: Failed to write results to {filename}")
            print(traceback.format_exc())
            raise

        return filename
