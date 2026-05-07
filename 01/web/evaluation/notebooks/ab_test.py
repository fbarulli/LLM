import sys
import os
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')

from src.search import CourseRAGManager
from src.config_manager import load_config
import pandas as pd

web_root = '/home/admin/LLM/LLM/01/web'

def get_datatalks_eval_set(num_queries=10):
    import json
    from src.core import generate_document_id
    
    with open(os.path.join(web_root, 'documents_datatalks.json'), 'r') as f:
        data = json.load(f)
    
    docs = data[0]['documents'][:num_queries]
    eval_set = []
    for doc in docs:
        doc_id = generate_document_id({
            'text': doc['text'],
            'question': doc['question'],
            'course': 'datatalks-zoomcamp'
        })
        eval_set.append({
            'query': doc['question'],
            'course': 'datatalks-zoomcamp',
            'expected_id': doc_id
        })
    return eval_set

def run_ab_test(config_a, config_b, num_queries=10):
    os.chdir(web_root)
    
    eval_set = get_datatalks_eval_set(num_queries)
    
    settings_a = load_config(f"experiments/configs/{config_a}.json")
    settings_b = load_config(f"experiments/configs/{config_b}.json")
    
    manager_a = CourseRAGManager(settings_a)
    manager_b = CourseRAGManager(settings_b)
    manager_a.connect_elasticsearch()
    manager_b.connect_elasticsearch()
    
    results = []
    for item in eval_set[:num_queries]:
        res_a = manager_a.search_faq(item['query'], 3, item['course'])
        res_b = manager_b.search_faq(item['query'], 3, item['course'])
        
        results.append({
            'query': item['query'][:60],
            'expected_course': item['course'],
            'config_a_course': res_a[0]['_source']['course'] if res_a else 'NONE',
            'config_a_score': round(res_a[0]['_score'], 2) if res_a else 0,
            'config_b_course': res_b[0]['_source']['course'] if res_b else 'NONE',
            'config_b_score': round(res_b[0]['_score'], 2) if res_b else 0,
        })
    
    df = pd.DataFrame(results)
    
    print("=" * 80)
    print(f"A/B TEST: {config_a} vs {config_b}")
    print("=" * 80)
    
    for _, row in df.iterrows():
        print(f"\nQ: {row['query']}")
        print(f"Expected: {row['expected_course']}")
        print(f"[A] {config_a}: {row['config_a_course']} (score: {row['config_a_score']})")
        print(f"[B] {config_b}: {row['config_b_course']} (score: {row['config_b_score']})")
        print("-" * 40)
    
    a_wins = sum(row['config_a_score'] > row['config_b_score'] for _, row in df.iterrows())
    b_wins = sum(row['config_b_score'] > row['config_a_score'] for _, row in df.iterrows())
    ties = len(df) - a_wins - b_wins
    
    print("\n" + "=" * 40)
    print("WINNER SUMMARY")
    print("=" * 40)
    print(f"Config A ({config_a}) wins: {a_wins}")
    print(f"Config B ({config_b}) wins: {b_wins}")
    print(f"Ties: {ties}")
    
    return df

if __name__ == "__main__":
    # Use your actual config names: bm25.json, vector.json, hybrid.json
    df = run_ab_test("bm25", "vector", num_queries=10)
