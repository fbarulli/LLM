import sys
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')
import json
import time
import litellm
from tqdm import tqdm
from src.search import CourseRAGManager
from src.config_manager import load_config

def batch_evaluate_queries(queries_data):
    """Evaluate multiple queries in a single LLM call with compressed input."""
    
    # Compress prompts - shorter context (150 chars instead of 300)
    batch_prompt = """Rate each query-response pair. Return JSON array with 'relevant' (bool) and 'faithful' (bool).

"""
    for i, q in enumerate(queries_data):
        batch_prompt += f"""{i+1}. Q:{q['query'][:80]} | A:{q['response'][:150]}\n"""
    
    batch_prompt += "\nReturn: [{\"relevant\": true/false, \"faithful\": true/false}, ...]"
    
    try:
        response = litellm.completion(
            model="nvidia_nim/nvidia/nemotron-mini-4b-instruct",  # Smaller, faster model
            messages=[{"role": "user", "content": batch_prompt}],
            temperature=0,
            max_tokens=300,
            timeout=60
        )
        
        result_text = response.choices[0].message.content
        import json
        import re
        json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        print(f"Batch error: {e}")
    
    return [{'faithful': False, 'relevant': False} for _ in queries_data]

def main():
    print("=" * 60)
    print("OPTIMIZED BATCH EVALUATION V2")
    print("=" * 60)
    
    with open('../experiments/hard_eval_set.json', 'r') as f:
        hard_eval = json.load(f)
    
    # Larger batch size (10 queries)
    test_items = []
    for item in hard_eval[:30]:
        for variant in item['paraphrased_queries'][:1]:
            test_items.append({
                'query': variant,
                'expected_id': item['expected_id'],
                'expected_course': item['expected_course']
            })
    
    print(f"📊 Testing {len(test_items)} queries")
    
    settings = load_config('../experiments/configs/baseline_bm25.json')
    manager = CourseRAGManager(settings)
    manager.connect_elasticsearch()
    
    queries_data = []
    for q in tqdm(test_items, desc="Retrieving"):
        results = manager.search_faq(q['query'], 5, q['expected_course'])
        if results:
            queries_data.append({
                'query': q['query'],
                'response': results[0]['_source']['text'][:150],  # Compressed
                'expected_id': q['expected_id'],
                'results': results
            })
    
    # Larger batch size = 10
    batch_size = 10
    all_evaluations = []
    
    print(f"\n📊 Batch evaluating {len(queries_data)} queries in groups of {batch_size}...")
    
    for i in range(0, len(queries_data), batch_size):
        batch = queries_data[i:i + batch_size]
        print(f"\n   Batch {i//batch_size + 1}/{(len(queries_data)+batch_size-1)//batch_size}...")
        
        start = time.time()
        evaluations = batch_evaluate_queries(batch)
        elapsed = time.time() - start
        print(f"   Completed in {elapsed:.1f}s ({elapsed/len(batch):.1f}s per query)")
        
        for j, eval_result in enumerate(evaluations):
            all_evaluations.append({
                **batch[j],
                'faithful': eval_result.get('faithful', False),
                'relevant': eval_result.get('relevant', False),
                'success': batch[j]['expected_id'] in [hit['_id'] for hit in batch[j]['results']]
            })
    
    total = len(all_evaluations)
    recall_success = sum(1 for r in all_evaluations if r.get('success'))
    faithful_count = sum(1 for r in all_evaluations if r.get('faithful'))
    relevant_count = sum(1 for r in all_evaluations if r.get('relevant'))
    
    print("\n" + "=" * 60)
    print("📊 RESULTS (V2 - Optimized)")
    print("=" * 60)
    print(f"Total queries: {total}")
    print(f"Recall@5: {recall_success}/{total} = {recall_success/total*100:.1f}%")
    print(f"Faithful: {faithful_count}/{total} = {faithful_count/total*100:.1f}%")
    print(f"Relevant: {relevant_count}/{total} = {relevant_count/total*100:.1f}%")

if __name__ == "__main__":
    main()
