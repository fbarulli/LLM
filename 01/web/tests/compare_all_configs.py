import sys
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')
import json
import time
import litellm
from tqdm import tqdm
from src.search import CourseRAGManager
from src.config_manager import load_config

def batch_evaluate_queries(queries_data):
    """Evaluate multiple queries in a single LLM call."""
    
    batch_prompt = """Rate each query-response pair. Return JSON array with 'relevant' (bool) and 'faithful' (bool).

"""
    for i, q in enumerate(queries_data):
        batch_prompt += f"""{i+1}. Q:{q['query'][:80]} | A:{q['response'][:150]}\n"""
    
    batch_prompt += "\nReturn: [{\"relevant\": true/false, \"faithful\": true/false}, ...]"
    
    try:
        response = litellm.completion(
            model="nvidia_nim/nvidia/nemotron-mini-4b-instruct",
            messages=[{"role": "user", "content": batch_prompt}],
            temperature=0,
            max_tokens=300,
            timeout=60
        )
        
        result_text = response.choices[0].message.content
        import re
        json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        print(f"Batch error: {e}")
    
    return [{'faithful': False, 'relevant': False} for _ in queries_data]

def evaluate_config(config_name, config_path, test_queries):
    print(f"\n{'='*60}")
    print(f"EVALUATING: {config_name}")
    print('='*60)
    
    settings = load_config(config_path)
    manager = CourseRAGManager(settings)
    manager.connect_elasticsearch()
    
    # Get search results
    queries_data = []
    for q in tqdm(test_queries, desc="Retrieving"):
        results = manager.search_faq(q['query'], 5, q['expected_course'])
        if results:
            queries_data.append({
                'query': q['query'],
                'response': results[0]['_source']['text'][:150],
                'expected_id': q['expected_id'],
                'results': results
            })
        else:
            queries_data.append({
                'query': q['query'],
                'response': '',
                'expected_id': q['expected_id'],
                'results': []
            })
    
    # Batch evaluate
    batch_size = 10
    all_evaluations = []
    
    for i in range(0, len(queries_data), batch_size):
        batch = queries_data[i:i + batch_size]
        evaluations = batch_evaluate_queries(batch)
        
        for j, eval_result in enumerate(evaluations):
            all_evaluations.append({
                **batch[j],
                'faithful': eval_result.get('faithful', False),
                'relevant': eval_result.get('relevant', False),
                'success': batch[j]['expected_id'] in [hit['_id'] for hit in batch[j]['results']]
            })
    
    # Calculate results
    total = len(all_evaluations)
    recall = sum(1 for r in all_evaluations if r.get('success')) / total * 100
    faithful = sum(1 for r in all_evaluations if r.get('faithful')) / total * 100
    relevant = sum(1 for r in all_evaluations if r.get('relevant')) / total * 100
    
    print(f"\n📊 {config_name} Results:")
    print(f"   Recall@5: {recall:.1f}%")
    print(f"   Faithful: {faithful:.1f}%")
    print(f"   Relevant: {relevant:.1f}%")
    
    return {'recall': recall, 'faithful': faithful, 'relevant': relevant}

def main():
    print("=" * 60)
    print("COMPARING BM25 vs VECTOR vs HYBRID")
    print("=" * 60)
    
    # Load hard eval set (all 90 queries)
    with open('../experiments/hard_eval_set.json', 'r') as f:
        hard_eval = json.load(f)
    
    # Flatten all queries
    test_queries = []
    for item in hard_eval:
        for variant in item['paraphrased_queries']:
            test_queries.append({
                'query': variant,
                'expected_id': item['expected_id'],
                'expected_course': item['expected_course']
            })
    
    print(f"\n📊 Total test queries: {len(test_queries)}")
    
    # Define configs to test
    configs = [
        ("BM25 (filtered)", "../experiments/configs/baseline_bm25.json"),
        ("Vector", "../settings.json"),
        ("Hybrid", "../experiments/configs/hybrid_bm25.json"),
    ]
    
    results = {}
    for name, path in configs:
        results[name] = evaluate_config(name, path, test_queries)
    
    # Print comparison table
    print("\n" + "=" * 60)
    print("📊 FINAL COMPARISON TABLE")
    print("=" * 60)
    print(f"\n{'Config':<20} {'Recall@5':<12} {'Faithful':<12} {'Relevant':<12}")
    print("-" * 56)
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['recall']:.1f}%{'':<6} {metrics['faithful']:.1f}%{'':<6} {metrics['relevant']:.1f}%")
    
    # Determine best configs
    print("\n🏆 BEST BY CATEGORY:")
    best_recall = max(results.items(), key=lambda x: x[1]['recall'])
    best_faithful = max(results.items(), key=lambda x: x[1]['faithful'])
    best_relevant = max(results.items(), key=lambda x: x[1]['relevant'])
    
    print(f"   Best Recall@5: {best_recall[0]} ({best_recall[1]['recall']:.1f}%)")
    print(f"   Best Faithful: {best_faithful[0]} ({best_faithful[1]['faithful']:.1f}%)")
    print(f"   Best Relevant: {best_relevant[0]} ({best_relevant[1]['relevant']:.1f}%)")

if __name__ == "__main__":
    main()
