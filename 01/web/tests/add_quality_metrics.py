import sys
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')
import json
from src.search import CourseRAGManager
from src.config_manager import load_config

def evaluate_results(result_file):
    """Add faithfulness/relevancy to existing results"""
    with open(result_file, 'r') as f:
        data = json.load(f)
    
    settings = load_config('../experiments/configs/llama_full.json')
    manager = CourseRAGManager(settings)
    manager.connect_elasticsearch()
    
    for result in data['results']:
        if result['k'] == 5 and result['found_id'] != 'NONE':
            # Get the actual response text
            response = result.get('best_text', '')
            query = result['query']
            context = [response]
            
            eval_result = manager.evaluate_response(query, response, context)
            result['faithful'] = eval_result.get('faithful')
            result['relevant'] = eval_result.get('relevant')
    
    # Save with quality metrics
    output_file = result_file.replace('.json', '_with_quality.json')
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved to {output_file}")

# Run on your hard eval results
evaluate_results('../experiments/results/baseline_bm25_hard_full.json')
