import sys
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')
import json
import time
from tqdm import tqdm
from src.search import CourseRAGManager
from src.config_manager import load_config

def batch_evaluate(config_name, config_path, max_queries=None):
    print(f"\n{'='*60}")
    print(f"EVALUATING: {config_name}")
    print('='*60)
    
    # Load hard eval set
    with open('../experiments/hard_eval_set.json', 'r') as f:
        hard_eval = json.load(f)
    
    # Flatten all queries
    test_queries = []
    for item in hard_eval:
        for variant in item['paraphrased_queries']:
            test_queries.append({
                'query': variant,
                'expected_id': item['expected_id'],
                'expected_course': item['expected_course'],
                'original_question': item['original_question']
            })
    
    if max_queries:
        test_queries = test_queries[:max_queries]
    
    print(f"📊 Testing {len(test_queries)} queries")
    
    # Initialize manager
    settings = load_config(config_path)
    manager = CourseRAGManager(settings)
    manager.connect_elasticsearch()
    
    results = []
    
    for q in tqdm(test_queries, desc="Processing"):
        search_results = manager.search_faq(q['query'], 5, q['expected_course'])
        
        if search_results:
            response_text = search_results[0]['_source']['text']
            context = [response_text]
            
            evaluation = manager.evaluate_response(q['query'], response_text, context)
            
            results.append({
                'query': q['query'],
                'original_question': q['original_question'],
                'expected_course': q['expected_course'],
                'response': response_text[:300],
                'faithful': evaluation.get('faithful'),
                'relevant': evaluation.get('relevant'),
                'recall_success': q['expected_id'] in [hit['_id'] for hit in search_results]
            })
        else:
            results.append({
                'query': q['query'],
                'original_question': q['original_question'],
                'expected_course': q['expected_course'],
                'response': 'NO RESULTS',
                'faithful': None,
                'relevant': None,
                'recall_success': False
            })
        
        time.sleep(0.5)  # Rate limit
    
    # Calculate summary
    total = len(results)
    recall_success = sum(1 for r in results if r.get('recall_success'))
    faithful_count = sum(1 for r in results if r.get('faithful') is True)
    relevant_count = sum(1 for r in results if r.get('relevant') is True)
    
    print(f"\n{'='*60}")
    print(f"RESULTS FOR {config_name}")
    print('='*60)
    print(f"Total queries: {total}")
    print(f"Recall@5: {recall_success}/{total} = {recall_success/total*100:.1f}%")
    print(f"Faithful: {faithful_count}/{total} = {faithful_count/total*100:.1f}%")
    print(f"Relevant: {relevant_count}/{total} = {relevant_count/total*100:.1f}%")
    
    return results

# Run on BM25 (filtered)
bm25_results = batch_evaluate(
    "BM25 (filtered)",
    "../experiments/configs/baseline_bm25.json",
    max_queries=30  # Limit to 30 for faster testing
)

print("\n" + "="*60)
print("✅ Batch evaluation complete")
print("="*60)
