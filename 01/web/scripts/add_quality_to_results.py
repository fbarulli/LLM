#!/usr/bin/env python
"""Add faithfulness and relevancy metrics to existing result files."""

import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.quality_evaluator import batch_evaluate_quality

def add_quality_to_file(result_path: str):
    """Add quality metrics to a single result file."""
    
    with open(result_path, 'r') as f:
        data = json.load(f)
    
    # Filter results for K=5 where we have a response
    results = [r for r in data['results'] if r.get('k') == 5 and r.get('found_id') != 'NONE']
    
    if not results:
        print(f"No valid results in {result_path}")
        return
    
    # Prepare for batch evaluation
    queries_data = [{
        'query': r['query'],
        'response': r.get('best_text', '')[:150]
    } for r in results]
    
    # Evaluate
    evaluations = batch_evaluate_quality(queries_data)
    
    # Add metrics to results
    for r, eval_result in zip(results, evaluations):
        r['faithful'] = eval_result.get('faithful', False)
        r['relevant'] = eval_result.get('relevant', False)
    
    # Save
    output_path = result_path.replace('.json', '_with_quality.json')
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Added quality metrics to {os.path.basename(output_path)}")
    return output_path

if __name__ == "__main__":
    results_dir = "experiments/results"
    
    for fname in os.listdir(results_dir):
        if fname.endswith('.json') and 'hard' in fname and 'quality' not in fname:
            add_quality_to_file(os.path.join(results_dir, fname))
