# /home/admin/LLM/LLM/01/web/eval/llm_judge_fixed.py

import sys
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')

import json
import random
import litellm
import re
from dotenv import load_dotenv
load_dotenv('/home/admin/LLM/LLM/01/web/configs/.env')

def evaluate_single_query(query, context, answer):
    prompt = f"""Evaluate if the ANSWER is faithful to the CONTEXT and relevant to the QUESTION.

QUESTION: {query}

CONTEXT: {context[:500]}

ANSWER: {answer[:300]}

Return ONLY valid JSON: {{"faithful": true/false, "relevant": true/false}}"""

    try:
        response = litellm.completion(
            model="nvidia_nim/nvidia/nemotron-mini-4b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=100
        )
        result_text = response.choices[0].message.content.strip()
        
        result = json.loads(result_text)
        return {
            'faithful': result.get('faithful', False),
            'relevant': result.get('relevant', False)
        }
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print(f"Raw response: {result_text[:200]}")
        return {'faithful': False, 'relevant': False}
    except Exception as e:
        print(f"Error: {e}")
        return {'faithful': False, 'relevant': False}

def evaluate_config(config_name, samples, num_queries=20):
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
    
    faithful_count = 0
    relevant_count = 0
    success_count = 0
    
    for i, query in enumerate(samples[:num_queries], 1):
        if query not in results_by_query:
            continue
        
        q_data = results_by_query[query]
        contexts = q_data['contexts']
        
        if not contexts:
            continue
        
        top_context = contexts[0]
        answer = top_context[:300]
        
        print(f"\n[{i}/{num_queries}] {query[:60]}...")
        
        result = evaluate_single_query(query, top_context, answer)
        
        faithful = result.get('faithful', False)
        relevant = result.get('relevant', False)
        
        print(f"  Success: {q_data['success']}")
        print(f"  Faithful: {faithful}")
        print(f"  Relevant: {relevant}")
        
        if q_data['success']:
            success_count += 1
        if faithful:
            faithful_count += 1
        if relevant:
            relevant_count += 1
    
    print(f"\n{config_name} Summary ({num_queries} queries):")
    print(f"  Retrieval success: {success_count}/{num_queries} ({success_count/num_queries*100:.1f}%)")
    print(f"  Faithful answers: {faithful_count}/{num_queries} ({faithful_count/num_queries*100:.1f}%)")
    print(f"  Relevant answers: {relevant_count}/{num_queries} ({relevant_count/num_queries*100:.1f}%)")

if __name__ == "__main__":
    with open('/home/admin/LLM/LLM/01/web/experiments/results/bm25_default.json', 'r') as f:
        data = json.load(f)
    all_queries = list(set(r['query'] for r in data['results'] if r['k'] == 5))
    samples = random.sample(all_queries, min(20, len(all_queries)))
    
    print(f"Testing {len(samples)} random queries\n")
    
    for config in ['bm25_default', 'vector_default', 'hybrid_default']:
        evaluate_config(config, samples, num_queries=10)
