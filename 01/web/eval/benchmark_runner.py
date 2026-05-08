# /home/admin/LLM/LLM/01/web/eval/benchmark_runner.py

import json
import time
import logging
import traceback
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

from src.search import CourseRAGManager
from src.config_manager import load_full_config
from eval.eval_set import get_eval_set_from_es


class BenchmarkRunner:
    
    def __init__(self, config_name: str, batch_size: int = 50):
        self.config_name = config_name
        self.settings = load_full_config(config_name)
        self.manager = CourseRAGManager(self.settings)
        self.manager.connect_elasticsearch()
        self.batch_size = batch_size
        self.default_k_values = [1, 3, 5, 10]
        self.default_top_k = max(self.default_k_values)
    
    def run_benchmark(self, k_values: List[int] = None) -> Dict[str, Any]:
        # Use provided k_values or fallback to default
        if k_values is None:
            k_values = self.default_k_values.copy()
        top_k = max(k_values)
        
        eval_set = get_eval_set_from_es()
        if not eval_set:
            raise ValueError("Eval set is empty – cannot run benchmark.")
        
        total_queries = len(eval_set)
        logger.info(f"Running batch benchmark for '{self.settings.get('name')}' on {total_queries} queries with k={k_values}, batch_size={self.batch_size}")
        
        queries_data = []
        for idx, item in enumerate(eval_set):
            query = item['original_doc'].get('question', '')
            if not query:
                logger.warning(f"Dropping eval set item {idx} – missing 'question' field")
                continue
            queries_data.append({
                'query': query,
                'course': item['original_doc'].get('course', ''),
                'expected_id': item['expected_id']
            })
        
        # Recompute total batches after filtering
        total_batches = (len(queries_data) + self.batch_size - 1) // self.batch_size
        all_results = []
        
        for batch_idx in range(0, len(queries_data), self.batch_size):
            batch = queries_data[batch_idx:batch_idx + self.batch_size]
            batch_num = (batch_idx // self.batch_size) + 1
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} queries)")
            
            queries = [item['query'] for item in batch]
            courses = [item['course'] for item in batch]
            
            start_time = time.time()
            
            unique_courses = list(set(courses))
            batch_results_by_course = {}
            
            for course in unique_courses:
                course_queries = [q for q, c in zip(queries, courses) if c == course]
                course_indices = [i for i, c in enumerate(courses) if c == course]
                
                if course_queries:
                    try:
                        hits_list = self.manager.batch_search_faq(course_queries, top_k, course)
                        for idx_course, hits in zip(course_indices, hits_list):
                            batch_results_by_course[idx_course] = hits
                    except Exception as e:
                        logger.error(f"Batch search failed for course {course}: {e}")
                        # Continue with empty hits for those queries
                        for idx_course in course_indices:
                            batch_results_by_course[idx_course] = []
            
            # Total wall‑clock time for the whole batch (all courses)
            batch_latency_ms = (time.time() - start_time) * 1000
            
            for idx, item in enumerate(batch):
                hits = batch_results_by_course.get(idx, [])
                expected_id = item['expected_id']
                
                # Precompute full contexts and hit IDs (all retrieved docs up to top_k)
                full_contexts = [hit['_source'].get('answer', '') for hit in hits]
                hit_ids = [hit['_id'] for hit in hits]
                
                for k in k_values:
                    # Take first k contexts/hits
                    sliced_contexts = full_contexts[:k]
                    sliced_hit_ids = hit_ids[:k]
                    success = expected_id in sliced_hit_ids
                    
                    # Determine found_id: the ID of the matching document within first k, or "NONE"
                    found_id = "NONE"
                    if success:
                        # Find the matching hit's ID (first occurrence)
                        for i, hit_id in enumerate(sliced_hit_ids):
                            if hit_id == expected_id:
                                found_id = hit_id
                                break
                    else:
                        # If not found, still record the top hit's ID? Original used hits[0] always.
                        # To keep backward compatibility, we'll use the top hit's ID if any.
                        found_id = hit_ids[0] if hit_ids else "NONE"
                    
                    all_results.append({
                        'k': k,
                        'query': item['query'],
                        'expected_id': expected_id,
                        'found_id': found_id,
                        'success': success,
                        'score': hits[0]['_score'] if hits else 0.0,
                        'latency_ms': round(batch_latency_ms / len(batch), 2),
                        'found_course': hits[0]['_source'].get('course', 'NONE') if hits else 'NONE',
                        'contexts': sliced_contexts
                    })
        
        output = {
            'metadata': {
                'name': self.settings.get('name'),
                'config_name': self.config_name,
                'settings': self.settings,
                'timestamp': datetime.now().isoformat(),
                'total_queries': len(queries_data),
                'k_values': k_values,
                'batch_size': self.batch_size,
                'total_batches': total_batches
            },
            'results': all_results
        }
        
        return output
    
    def save_results(self, output: Dict[str, Any]) -> str:
        web_root = '/home/admin/LLM/LLM/01/web'
        results_dir = f'{web_root}/experiments/results'
        os.makedirs(results_dir, exist_ok=True)
        filename = f'{results_dir}/{self.config_name}.json'
        try:
            with open(filename, 'w') as f:
                json.dump(output, f, indent=2)
            logger.info(f"Saved results to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise
    
    def run_and_save(self, k_values: List[int] = None) -> str:
        results = self.run_benchmark(k_values)
        return self.save_results(results)


def run_benchmark(config_name: str, k_values: List[int] = None, batch_size: int = 50) -> str:
    """Standalone convenience function (kept for backward compatibility)."""
    runner = BenchmarkRunner(config_name, batch_size=batch_size)
    return runner.run_and_save(k_values)