# /home/admin/LLM/LLM/01/web/eval/benchmark_runner.py

import json
import time
import logging
import traceback
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
        self.k_values = [1, 3, 5, 10]
        self.top_k = max(self.k_values)
    
    def run_benchmark(self, k_values: List[int] = None) -> Dict[str, Any]:
        if k_values:
            self.k_values = k_values
            self.top_k = max(self.k_values)
        
        eval_set = get_eval_set_from_es()
        total_queries = len(eval_set)
        logger.info(f"Running batch benchmark for '{self.settings.get('name')}' on {total_queries} queries with k={self.k_values}, batch_size={self.batch_size}")
        
        queries_data = []
        for item in eval_set:
            query = item['original_doc'].get('question', '')
            if not query:
                continue
            queries_data.append({
                'query': query,
                'course': item['original_doc'].get('course', ''),
                'expected_id': item['expected_id']
            })
        
        all_results = []
        total_batches = (len(queries_data) + self.batch_size - 1) // self.batch_size
        
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
                    hits_list = self.manager.batch_search_faq(course_queries, self.top_k, course)
                    
                    for idx, hits in zip(course_indices, hits_list):
                        batch_results_by_course[idx] = hits
            
            latency_ms = (time.time() - start_time) * 1000
            
            for idx, item in enumerate(batch):
                hits = batch_results_by_course.get(idx, [])
                hit_ids = [hit['_id'] for hit in hits]
                top_hit = hits[0] if hits else None
                expected_id = item['expected_id']
                contexts = [hit['_source'].get('text', '') for hit in hits] if hits else []
                
                for k in self.k_values:
                    success = expected_id in hit_ids[:k]
                    all_results.append({
                        'k': k,
                        'query': item['query'],
                        'expected_id': expected_id,
                        'found_id': top_hit['_id'] if top_hit else 'NONE',
                        'success': success,
                        'score': top_hit['_score'] if top_hit else 0.0,
                        'latency_ms': round(latency_ms / len(batch), 2),
                        'found_course': top_hit['_source'].get('course', 'NONE') if top_hit else 'NONE',
                        'contexts': contexts
                    })
        
        output = {
            'metadata': {
                'name': self.settings.get('name'),
                'config_name': self.config_name,
                'settings': self.settings,
                'timestamp': datetime.now().isoformat(),
                'total_queries': len(queries_data),
                'k_values': self.k_values,
                'batch_size': self.batch_size,
                'total_batches': total_batches
            },
            'results': all_results
        }
        
        return output
    
    def save_results(self, output: Dict[str, Any]) -> str:
        web_root = '/home/admin/LLM/LLM/01/web'
        results_dir = f'{web_root}/experiments/results'
        
        import os
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
    runner = BenchmarkRunner(config_name, batch_size=batch_size)
    return runner.run_and_save(k_values)