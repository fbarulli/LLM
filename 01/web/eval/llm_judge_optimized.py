# /home/admin/LLM/LLM/01/web/eval/llm_judge_optimized.py

import sys
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')

import json
import random
import litellm
import re
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
load_dotenv('/home/admin/LLM/LLM/01/web/configs/.env')

CACHE_FILE = '/home/admin/LLM/LLM/01/web/experiments/llm_cache.json'

def load_cache():
    try:
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    except:
        return {}

def save_cache(cache):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)

def batch_evaluate_llm_parallel(queries_data, batch_size=10, max_workers=2):
    cache = load_cache()
    uncached_items = []
    cached_results = []
    
    for q in queries_data:
        cache_key = hashlib.md5(f"{q['query']}{q.get('context', '')[:500]}{q.get('answer', '')[:200]}".encode()).hexdigest()
        if cache_key in cache:
            cached_results.append(cache[cache_key])
        else:
            uncached_items.append((cache_key, q))
    
    if not uncached_items:
        return cached_results
    
    def process_batch(batch_items):
        batch = [item[1] for item in batch_items]
        cache_keys = [item[0] for item in batch_items]
        
        prompt = """Rate if answer is faithful (uses context) and relevant (answers question). Return JSON array.

"""
        for j, q in enumerate(batch):
            context_full = q['context'][:500] if q['context'] else "No context"
            prompt += f"""{j+1}. Question: {q['query']}
Context: {context_full}...
Answer: {q['answer'][:300]}

"""
        
        prompt += 'Return: [{"faithful": bool, "relevant": bool}]'
        
        try:
            response = litellm.completion(
                model="nvidia_nim/nvidia/nemotron-mini-4b-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=500
            )
            result_text = response.choices[0].message.content
            json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
            if json_match:
                batch_results = json.loads(json_match.group())
                for cache_key, result in zip(cache_keys, batch_results):
                    cache[cache_key] = result
                return batch_results
        except Exception as e:
            print(f"Batch error: {e}")
        
        default_results = [{'faithful': False, 'relevant': False} for _ in batch]
        for cache_key, result in zip(cache_keys, default_results):
            cache[cache_key] = result
        return default_results
    
    batches = [uncached_items[i:i+batch_size] for i in range(0, len(uncached_items), batch_size)]
    
    batch_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_batch, batch) for batch in batches]
        for future in as_completed(futures):
            batch_results.extend(future.result())
    
    save_cache(cache)
    
    return cached_results + batch_results

def evaluate_config_optimized(config_name, samples, num_queries=100, batch_size=10):
    print(f"\n{'='*50}")
    print(f"Evaluating: {config_name}")
    print('='*50)
    
    with open(f'/home/admin/LLM/LLM/01/web/experiments/results/{config_name}.json', 'r') as f:
        data = json.load(f)
    
    results_by_query = {}
    for r in data['results']:
        if r['k'] == 5:
            results_by_query[r['query']] = {
                'success': r['success'],
                'contexts': r.get('contexts', []),
                'found_id': r['found_id']
            }
    
    queries_data = []
    for query in samples[:num_queries]:
        if query in results_by_query and results_by_query[query]['contexts']:
            contexts = results_by_query[query]['contexts']
            top_context = contexts[0] if contexts else ""
            queries_data.append({
                'query': query,
                'context': top_context,
                'answer': top_context[:500] if top_context else "",
                'success': results_by_query[query]['success']
            })
    
    print(f"Evaluating {len(queries_data)} queries (batch_size={batch_size}, workers={2})...")
    eval_results = batch_evaluate_llm_parallel(queries_data, batch_size=batch_size, max_workers=2)
    
    faithful_count = sum(1 for r in eval_results if r.get('faithful'))
    relevant_count = sum(1 for r in eval_results if r.get('relevant'))
    success_count = sum(1 for q in queries_data if q['success'])
    
    print(f"\n{config_name} Results ({len(queries_data)} queries):")
    print(f"  Retrieval success: {success_count}/{len(queries_data)} ({success_count/len(queries_data)*100:.1f}%)")
    print(f"  Faithful answers: {faithful_count}/{len(queries_data)} ({faithful_count/len(queries_data)*100:.1f}%)")
    print(f"  Relevant answers: {relevant_count}/{len(queries_data)} ({relevant_count/len(queries_data)*100:.1f}%)")
    
    return faithful_count, relevant_count, len(queries_data)

if __name__ == "__main__":
    with open('/home/admin/LLM/LLM/01/web/experiments/results/bm25_default.json', 'r') as f:
        data = json.load(f)
    all_queries = list(set(r['query'] for r in data['results'] if r['k'] == 5))
    samples = random.sample(all_queries, min(100, len(all_queries)))
    
    print(f"Testing {len(samples)} random queries across configs\n")
    
    configs = ['bm25_default', 'vector_default', 'hybrid_default']
    
    for config in configs:
        evaluate_config_optimized(config, samples, num_queries=100, batch_size=10)
