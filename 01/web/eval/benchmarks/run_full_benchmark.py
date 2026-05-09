#!/usr/bin/env python

import sys
import os
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')

import time
import json
import argparse
from typing import List, Dict, Any

from eval.benchmark_runner import run_benchmark
from eval.qdrant_benchmark_runner import run_qdrant_benchmark


CONFIGS = {
    'bm25_default': 'BM25 Default',
    'bm25_balanced': 'BM25 Balanced',
    'bm25_high_question': 'BM25 High Question',
    'bm25_high_text': 'BM25 High Text',
    'vector_default': 'Vector Default',
    'hybrid_default': 'Hybrid Default',
    'hybrid_balanced': 'Hybrid Balanced',
    'qdrant_default': 'Qdrant Vector Default',
}


def run_all_benchmarks(batch_size: int = 50) -> List[Dict[str, Any]]:
    results = []
    total_start = time.time()
    
    print("=" * 60)
    print("RUNNING FULL BENCHMARK - ALL CONFIGS")
    print("=" * 60)
    print(f"Batch size: {batch_size}")
    print(f"Configs to run: {len(CONFIGS)}\n")
    
    for config_name, display_name in CONFIGS.items():
        start = time.time()
        print(f"[{config_name}] {display_name}...")
        
        try:
            if config_name == 'qdrant_default':
                filename = run_qdrant_benchmark(batch_size=batch_size)
            else:
                filename = run_benchmark(config_name, batch_size=batch_size)
            
            elapsed = time.time() - start
            
            with open(filename, 'r') as f:
                data = json.load(f)
                total_queries = data['metadata']['total_queries']
                results_data = data['results']
                
                recall_at_5 = sum(1 for r in results_data if r['k'] == 5 and r['success']) / total_queries * 100
                
            results.append({
                'config': config_name,
                'display_name': display_name,
                'time': round(elapsed, 2),
                'recall_at_5': round(recall_at_5, 1),
                'file': filename
            })
            
            print(f"  Done in {elapsed:.2f}s | Recall@5: {recall_at_5:.1f}%\n")
            
        except Exception as e:
            print(f"  Failed: {e}\n")
            results.append({
                'config': config_name,
                'display_name': display_name,
                'error': str(e)
            })
    
    total_elapsed = time.time() - total_start
    
    print("=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Total time: {total_elapsed:.2f} seconds\n")
    
    for r in results:
        if 'recall_at_5' in r:
            print(f"  {r['display_name']:25}: {r['recall_at_5']}% ({r['time']}s)")
        else:
            print(f"  {r['display_name']:25}: FAILED")
    
    print("\nResults saved to: /home/admin/LLM/LLM/01/web/experiments/results/")
    
    return results


def run_single_benchmark(config_name: str, batch_size: int = 50) -> Dict[str, Any]:
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Choose from: {list(CONFIGS.keys())}")
    
    print(f"Running {CONFIGS[config_name]}...")
    start = time.time()
    
    if config_name == 'qdrant_default':
        filename = run_qdrant_benchmark(batch_size=batch_size)
    else:
        filename = run_benchmark(config_name, batch_size=batch_size)
    
    elapsed = time.time() - start
    
    with open(filename, 'r') as f:
        data = json.load(f)
        total_queries = data['metadata']['total_queries']
        results_data = data['results']
        
        recall_at_5 = sum(1 for r in results_data if r['k'] == 5 and r['success']) / total_queries * 100
    
    return {
        'config': config_name,
        'display_name': CONFIGS[config_name],
        'time': round(elapsed, 2),
        'recall_at_5': round(recall_at_5, 1),
        'file': filename
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run benchmarks for search configs')
    parser.add_argument('--config', type=str, help='Run only a specific config')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for ES queries')
    
    args = parser.parse_args()
    
    if args.config:
        result = run_single_benchmark(args.config, batch_size=args.batch_size)
        print(f"\nResult: {result['display_name']} - Recall@5: {result['recall_at_5']}% in {result['time']}s")
    else:
        run_all_benchmarks(batch_size=args.batch_size)
