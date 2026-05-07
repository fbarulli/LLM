# /home/admin/LLM/LLM/01/web/eval/llm_judge_batched.py

import sys
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')

import json
import random
import litellm
from dotenv import load_dotenv
load_dotenv('/home/admin/LLM/LLM/01/web/configs/.env')

from src.search import CourseRAGManager
from src.config_manager import load_full_config
from eval.eval_set import get_eval_set_from_es

def batch_evaluate_llm(queries_data, batch_size=5):
    results = []
    
    for i in range(0, len(queries_data), batch_size):
        batch = queries_data[i:i+batch_size]
        
        prompt = """Rate each query-response pair. Return JSON array with 'relevant' (bool) and 'faithful' (bool).

"""
        for j, q in enumerate(batch):
            prompt += f"""{j+1}. Question: {q['query'][:100]}\n   Answer: {q['response'][:200]}\n\n"""
        
        prompt += 'Return: [{"relevant": true/false, "faithful": true/false}, ...]'
        
        try:
            response = litellm.completion(
                model="nvidia_nim/nvidia/nemotron-mini-4b-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=500
            )
            result_text = response.choices[0].message.content
            import re
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

def evaluate_config_batched(config_name, samples, num_queries=10):
    print(f"\n{'='*50}")
    print(f"Evaluating: {config_name}")
    print('='*50)
    
    config = load_full_config(config_name)
    manager = CourseRAGManager(config)
    manager.connect_elasticsearch()
    
    queries_data = []
    
    for item in samples[:num_queries]:
        query = item['original_doc'].get('question', '')
        hits = manager.search_faq(query, 3, None)
        
        if hits:
            response_text = hits[0]['_source'].get('text', '')
            queries_data.append({
                'query': query,
                'response': response_text
            })
    
    print(f"Evaluating {len(queries_data)} queries in batches...")
    eval_results = batch_evaluate_llm(queries_data, batch_size=5)
    
    for i, (qd, er) in enumerate(zip(queries_data, eval_results), 1):
        print(f"\n[{i}] {qd['query'][:60]}...")
        print(f"  Faithful: {er.get('faithful')}")
        print(f"  Relevant: {er.get('relevant')}")
    
    return eval_results

if __name__ == "__main__":
    eval_set = get_eval_set_from_es()
    samples = random.sample(eval_set, 10)
    
    configs = ['bm25_default', 'vector_default', 'hybrid_default']
    
    for config in configs:
        results = evaluate_config_batched(config, samples, num_queries=10)
        faithful = sum(1 for r in results if r.get('faithful') is True)
        relevant = sum(1 for r in results if r.get('relevant') is True)
        
        print(f"\n{config} Summary:")
        print(f"  Faithful: {faithful}/{len(results)}")
        print(f"  Relevant: {relevant}/{len(results)}")
