import sys
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')
import json
import time
import litellm
from tqdm import tqdm
from src.search import CourseRAGManager
from src.config_manager import load_config

# Enable litellm debugging
litellm.set_verbose = True

def batch_evaluate_queries(queries_data, batch_num):
    """Evaluate multiple queries in a single LLM call."""
    
    print(f"\n[DEBUG] Batch {batch_num}: Building prompt for {len(queries_data)} queries")
    
    batch_prompt = """You are evaluating a search system. For each query-response pair, determine if the response answers the question (relevant) and if it's truthful to the context (faithful).

Return ONLY a JSON array. Each item must have 'relevant' (boolean) and 'faithful' (boolean).

"""
    for i, q in enumerate(queries_data):
        batch_prompt += f"""
Item {i+1}:
QUESTION: {q['query']}
CONTEXT: {q['context'][:200]}
RESPONSE: {q['response'][:200]}
---"""
    
    batch_prompt += "\n\nJSON array: [{\"relevant\": true/false, \"faithful\": true/false}, ...]"
    
    print(f"[DEBUG] Prompt length: {len(batch_prompt)} chars")
    print(f"[DEBUG] Sending to NVIDIA API...")
    
    try:
        start = time.time()
        response = litellm.completion(
            model="nvidia_nim/meta/llama-3.1-8b-instruct",
            messages=[{"role": "user", "content": batch_prompt}],
            temperature=0,
            max_tokens=500,
            timeout=120
        )
        elapsed = time.time() - start
        print(f"[DEBUG] API call completed in {elapsed:.1f}s")
        print(f"[DEBUG] Raw response: {response.choices[0].message.content[:200]}...")
        
        import json
        import re
        result_text = response.choices[0].message.content
        json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            print(f"[DEBUG] Parsed {len(parsed)} results")
            return parsed
        else:
            print(f"[DEBUG] No JSON array found in response")
            return [{'faithful': False, 'relevant': False} for _ in queries_data]
            
    except Exception as e:
        print(f"[ERROR] Batch evaluation failed: {e}")
        return [{'faithful': False, 'relevant': False} for _ in queries_data]

def main():
    print("=" * 60)
    print("OPTIMIZED BATCH EVALUATION (DEBUG MODE)")
    print("=" * 60)
    
    # Load hard eval set
    with open('../experiments/hard_eval_set.json', 'r') as f:
        hard_eval = json.load(f)
    
    # Prepare queries (first 10 for testing)
    test_items = []
    for item in hard_eval[:10]:
        for variant in item['paraphrased_queries'][:1]:
            test_items.append({
                'query': variant,
                'expected_id': item['expected_id'],
                'expected_course': item['expected_course']
            })
    
    print(f"📊 Testing {len(test_items)} queries")
    
    # Get search results first (fast)
    settings = load_config('../experiments/configs/baseline_bm25.json')
    manager = CourseRAGManager(settings)
    manager.connect_elasticsearch()
    
    queries_data = []
    for q in tqdm(test_items, desc="Retrieving"):
        results = manager.search_faq(q['query'], 5, q['expected_course'])
        if results:
            queries_data.append({
                'query': q['query'],
                'context': results[0]['_source']['text'][:300],
                'response': results[0]['_source']['text'][:300],
                'expected_id': q['expected_id'],
                'results': results
            })
    
    # Batch evaluate in groups of 5
    batch_size = 5
    all_evaluations = []
    
    print(f"\n📊 Batch evaluating {len(queries_data)} queries in groups of {batch_size}...")
    
    for i in range(0, len(queries_data), batch_size):
        batch = queries_data[i:i + batch_size]
        batch_num = i//batch_size + 1
        total_batches = (len(queries_data) + batch_size - 1) // batch_size
        
        print(f"\n{'='*40}")
        print(f"Batch {batch_num}/{total_batches}")
        print(f"{'='*40}")
        
        evaluations = batch_evaluate_queries(batch, batch_num)
        
        print(f"[DEBUG] Evaluations received: {len(evaluations)}")
        for j, eval_result in enumerate(evaluations):
            print(f"  Item {j+1}: faithful={eval_result.get('faithful', '?')}, relevant={eval_result.get('relevant', '?')}")
            all_evaluations.append({
                **batch[j],
                'faithful': eval_result.get('faithful', False),
                'relevant': eval_result.get('relevant', False),
                'success': batch[j]['expected_id'] in [hit['_id'] for hit in batch[j]['results']]
            })
    
    # Calculate results
    total = len(all_evaluations)
    recall_success = sum(1 for r in all_evaluations if r.get('success'))
    faithful_count = sum(1 for r in all_evaluations if r.get('faithful'))
    relevant_count = sum(1 for r in all_evaluations if r.get('relevant'))
    
    print("\n" + "=" * 60)
    print("📊 RESULTS (Optimized Batch)")
    print("=" * 60)
    print(f"Total queries: {total}")
    print(f"Recall@5: {recall_success}/{total} = {recall_success/total*100:.1f}%")
    print(f"Faithful: {faithful_count}/{total} = {faithful_count/total*100:.1f}%")
    print(f"Relevant: {relevant_count}/{total} = {relevant_count/total*100:.1f}%")
    
    # Show failures
    print("\n❌ FAILURES:")
    failures = [r for r in all_evaluations if not r.get('faithful') or not r.get('relevant')]
    for r in failures[:3]:
        print(f"  - Query: {r['query'][:60]}...")
        print(f"    Faithful: {r.get('faithful')}, Relevant: {r.get('relevant')}")

if __name__ == "__main__":
    main()
