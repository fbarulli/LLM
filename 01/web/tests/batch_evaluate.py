import sys
import json
import time
from tqdm import tqdm
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')

from src.search import CourseRAGManager
from src.config_manager import load_config

def evaluate_hard_set(config_name: str, config_path: str, max_queries: int = 30):
    print(f"\n{'='*60}")
    print(f"EVALUATING: {config_name}")
    print('='*60)
    
    # Load hard eval set
    with open('../experiments/hard_eval_set.json', 'r') as f:
        hard_eval = json.load(f)
    
    # Flatten queries (limit to max_queries)
    test_queries = []
    for item in hard_eval[:max_queries//3]:
        for variant in item['paraphrased_queries'][:1]:
            test_queries.append({
                'query': variant,
                'expected_id': item['expected_id'],
                'expected_course': item['expected_course'],
                'original_question': item['original_question']
            })
    
    print(f"📊 Testing {len(test_queries)} queries")
    
    # Initialize manager
    settings = load_config(config_path)
    manager = CourseRAGManager(settings)
    manager.connect_elasticsearch()
    
    results = []
    
    for q in tqdm(test_queries, desc="Processing"):
        # Search
        search_results = manager.search_faq(q['query'], 3, q['expected_course'])
        
        if search_results:
            response_text = search_results[0]['_source']['text']
            context = [response_text]
            
            # Evaluate faithfulness and relevancy
            evaluation = manager.evaluate_response(q['query'], response_text, context)
            
            results.append({
                'query': q['query'],
                'original_question': q['original_question'],
                'expected_course': q['expected_course'],
                'response': response_text[:200],
                'faithful': evaluation.get('faithful'),
                'faithfulness_score': evaluation.get('faithfulness_score'),
                'relevant': evaluation.get('relevant'),
                'relevancy_score': evaluation.get('relevancy_score'),
            })
        else:
            results.append({
                'query': q['query'],
                'original_question': q['original_question'],
                'expected_course': q['expected_course'],
                'response': 'NO RESULTS',
                'faithful': None,
                'faithfulness_score': None,
                'relevant': None,
                'relevancy_score': None,
            })
        
        time.sleep(0.5)  # Rate limit
    
    # Calculate summary
    faithful_count = sum(1 for r in results if r.get('faithful') is True)
    relevant_count = sum(1 for r in results if r.get('relevant') is True)
    
    print(f"\n{'='*60}")
    print(f"RESULTS FOR {config_name}")
    print('='*60)
    print(f"Total queries: {len(results)}")
    print(f"Faithful responses: {faithful_count}/{len(results)} = {faithful_count/len(results)*100:.1f}%")
    print(f"Relevant responses: {relevant_count}/{len(results)} = {relevant_count/len(results)*100:.1f}%")
    
    # Show failures
    print("\n❌ FAILURES:")
    failures = [r for r in results if not r.get('faithful') or not r.get('relevant')]
    for r in failures[:5]:
        print(f"  - Query: {r['query'][:60]}...")
        print(f"    Faithful: {r.get('faithful')}, Relevant: {r.get('relevant')}")
    
    return results

# Run evaluations on both configs
print("="*60)
print("BATCH EVALUATION ON HARD EVAL SET")
print("="*60)

# BM25 results
bm25_results = evaluate_hard_set(
    "BM25", 
    "../experiments/configs/baseline_bm25.json",
    max_queries=30
)

# Vector results
vector_results = evaluate_hard_set(
    "VECTOR", 
    "../settings.json",
    max_queries=30
)

# Compare
print("\n" + "="*60)
print("COMPARISON SUMMARY")
print("="*60)

bm25_faithful = sum(1 for r in bm25_results if r.get('faithful') is True) / len(bm25_results) * 100
bm25_relevant = sum(1 for r in bm25_results if r.get('relevant') is True) / len(bm25_results) * 100
vector_faithful = sum(1 for r in vector_results if r.get('faithful') is True) / len(vector_results) * 100
vector_relevant = sum(1 for r in vector_results if r.get('relevant') is True) / len(vector_results) * 100

print(f"\n{'Config':<15} {'Faithful':<12} {'Relevant':<12}")
print(f"{'-'*40}")
print(f"{'BM25':<15} {bm25_faithful:.1f}%{'':<6} {bm25_relevant:.1f}%")
print(f"{'VECTOR':<15} {vector_faithful:.1f}%{'':<6} {vector_relevant:.1f}%")

# Save results
with open('../experiments/evaluation_results.json', 'w') as f:
    json.dump({
        'bm25': bm25_results,
        'vector': vector_results,
        'summary': {
            'bm25_faithful': bm25_faithful,
            'bm25_relevant': bm25_relevant,
            'vector_faithful': vector_faithful,
            'vector_relevant': vector_relevant
        }
    }, f, indent=2)

print("\n✅ Results saved to experiments/evaluation_results.json")
