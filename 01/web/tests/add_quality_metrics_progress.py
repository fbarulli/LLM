import sys
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')
import json
from tqdm import tqdm
from src.search import CourseRAGManager
from src.config_manager import load_config

def evaluate_results(result_file):
    print(f"Loading results from {result_file}...")
    with open(result_file, 'r') as f:
        data = json.load(f)
    
    print("Initializing LLM evaluators...")
    settings = load_config('../experiments/configs/llama_full.json')
    manager = CourseRAGManager(settings)
    manager.connect_elasticsearch()
    
    # Filter results that need evaluation
    to_evaluate = [r for r in data['results'] 
                   if r['k'] == 5 and r['found_id'] != 'NONE']
    
    print(f"Evaluating {len(to_evaluate)} results...")
    
    for result in tqdm(to_evaluate, desc="Processing"):
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
    print(f"\n✅ Saved to {output_file}")
    
    # Print summary
    faithful_count = sum(1 for r in to_evaluate if r.get('faithful') is True)
    relevant_count = sum(1 for r in to_evaluate if r.get('relevant') is True)
    print(f"\n📊 Summary:")
    print(f"   Faithful: {faithful_count}/{len(to_evaluate)} = {faithful_count/len(to_evaluate)*100:.1f}%")
    print(f"   Relevant: {relevant_count}/{len(to_evaluate)} = {relevant_count/len(to_evaluate)*100:.1f}%")

if __name__ == "__main__":
    evaluate_results('../experiments/results/baseline_bm25_hard_full.json')
