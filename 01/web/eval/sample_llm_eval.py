# /home/admin/LLM/LLM/01/web/eval/sample_llm_eval.py

import sys
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')

import random
from dotenv import load_dotenv
load_dotenv('/home/admin/LLM/LLM/01/web/configs/.env')

from src.search import CourseRAGManager
from src.config_manager import load_full_config
from eval.eval_set import get_eval_set_from_es

def evaluate_config(config_name, samples, num_queries=10):
    print(f"\n{'='*50}")
    print(f"Evaluating: {config_name}")
    print('='*50)
    
    config = load_full_config(config_name)
    manager = CourseRAGManager(config)
    manager.connect_elasticsearch()
    
    results = []
    
    for i, item in enumerate(samples[:num_queries], 1):
        query = item['original_doc'].get('question', '')
        expected_id = item['expected_id']
        
        print(f"\n[{i}/{num_queries}] {query[:60]}...")
        
        hits = manager.search_faq(query, 3, None)
        
        if hits:
            contexts = [hit['_source'].get('text', '') for hit in hits]
            response_text = contexts[0] if contexts else ""
            
            triad = manager.evaluate_rag_triad(query, response_text, contexts)
            
            success = expected_id in [hit['_id'] for hit in hits]
            
            results.append({
                'query': query[:80],
                'success': success,
                'faithful': triad.get('faithful'),
                'relevant': triad.get('relevant')
            })
            
            print(f"  Success: {success}")
            print(f"  Faithful: {triad.get('faithful')}")
            print(f"  Relevant: {triad.get('relevant')}")
        else:
            results.append({'query': query[:80], 'success': False, 'faithful': None, 'relevant': None})
            print(f"  No results")
    
    return results

if __name__ == "__main__":
    eval_set = get_eval_set_from_es()
    samples = random.sample(eval_set, 10)
    
    configs = ['bm25_default', 'vector_default', 'hybrid_default']
    
    all_results = {}
    for config in configs:
        results = evaluate_config(config, samples, num_queries=10)
        all_results[config] = results
        
        faithful_count = sum(1 for r in results if r.get('faithful') is True)
        relevant_count = sum(1 for r in results if r.get('relevant') is True)
        success_count = sum(1 for r in results if r.get('success') is True)
        
        print(f"\n{config} Summary:")
        print(f"  Retrieval success: {success_count}/10")
        print(f"  Faithful answers: {faithful_count}/10")
        print(f"  Relevant answers: {relevant_count}/10")
    
    print("\n" + "="*50)
    print("COMPARISON SUMMARY")
    print("="*50)
    for config in configs:
        results = all_results[config]
        print(f"\n{config}:")
        print(f"  Success: {sum(1 for r in results if r['success'])}/10")
        print(f"  Faithful: {sum(1 for r in results if r.get('faithful') is True)}/10")
        print(f"  Relevant: {sum(1 for r in results if r.get('relevant') is True)}/10")
