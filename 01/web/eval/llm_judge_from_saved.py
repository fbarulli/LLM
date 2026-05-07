# /home/admin/LLM/LLM/01/web/eval/llm_judge_from_saved.py

import sys
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')

import json
import random
import litellm
import re
from dotenv import load_dotenv
load_dotenv('/home/admin/LLM/LLM/01/web/configs/.env')

def load_results_with_contexts(config_name):
    with open(f'/home/admin/LLM/LLM/01/web/experiments/results/{config_name}.json', 'r') as f:
        data = json.load(f)
    
    results_by_query = {}
    for r in data['results']:
        if r['k'] == 5:
            results_by_query[r['query']] = {
                'success': r['success'],
                'found_id': r['found_id'],
                'contexts': r.get('contexts', [])
            }
    return results_by_query

def batch_evaluate_llm(queries_data, batch_size=5):
    results = []
    
    for i in range(0, len(queries_data), batch_size):
        batch = queries_data[i:i+batch_size]
        
        prompt = """Rate each query-context-response pair. Return JSON array with 'relevant' (bool) and 'faithful' (bool).

"""
        for j, q in enumerate(batch):
            contexts_preview = q['contexts'][0][:150] if q['contexts'] else "No context"
            prompt += f"""{j+1}. Question: {q['query'][:80]}
   Context: {contexts_preview}...
   Answer: Uses context directly

"""
        
        prompt += 'Return: [{"relevant": true/false, "faithful": true/false}, ...]'
        
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
                results.extend(batch_results)
            else:
                results.extend([{'faithful': False, 'relevant': False} for _ in batch])
        except Exception as e:
            print(f"Batch error: {e}")
            results.extend([{'faithful': False, 'relevant': False} for _ in batch])
    
    return results

def evaluate_config_from_saved(config_name, samples, num_queries=10):
    print(f"\n{'='*50}")
    print(f"Evaluating: {config_name}")
    print('='*50)
    
    results_by_query = load_results_with_contexts(config_name)
    
    queries_data = []
    for query in samples[:num_queries]:
        if query in results_by_query:
            queries_data.append({
                'query': query,
                'contexts': results_by_query[query]['contexts'],
                'success': results_by_query[query]['success']
            })
    
    print(f"Evaluating {len(queries_data)} queries from saved results...")
    eval_results = batch_evaluate_llm(queries_data, batch_size=5)
    
    faithful_count = 0
    relevant_count = 0
    
    for i, (qd, er) in enumerate(zip(queries_data, eval_results), 1):
        print(f"\n[{i}] {qd['query'][:60]}...")
        print(f"  Success: {qd['success']}")
        print(f"  Faithful: {er.get('faithful')}")
        print(f"  Relevant: {er.get('relevant')}")
        if er.get('faithful'):
            faithful_count += 1
        if er.get('relevant'):
            relevant_count += 1
    
    return faithful_count, relevant_count, len(queries_data)

if __name__ == "__main__":
    eval_set = json.load(open('/home/admin/LLM/LLM/01/web/experiments/results/bm25_default.json'))
    all_queries = list(set(r['query'] for r in eval_set['results'] if r['k'] == 5))
    samples = random.sample(all_queries, 10)
    
    configs = ['bm25_default', 'vector_default', 'hybrid_default']
    
    print(f"Testing {len(samples)} random queries across configs\n")
    
    for config in configs:
        faithful, relevant, total = evaluate_config_from_saved(config, samples, num_queries=10)
        print(f"\n{config} Summary:")
        print(f"  Faithful: {faithful}/{total}")
        print(f"  Relevant: {relevant}/{total}")
