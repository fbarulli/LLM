# /home/admin/LLM/LLM/01/web/notebooks/ab_test.py

import sys
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')

from src.search import CourseRAGManager
from src.config_manager import load_config
from src.run_stats import get_eval_set
import pandas as pd
import os

web_root = '/home/admin/LLM/LLM/01/web'

def run_ab_test(config_a, config_b, num_queries=10):
    os.chdir(web_root)
    
    eval_set = get_eval_set("documents.json", n_per_course=num_queries // 3)
    
    settings_a = load_config(f"experiments/configs/{config_a}.json")
    settings_b = load_config(f"experiments/configs/{config_b}.json")
    
    manager_a = CourseRAGManager(settings_a)
    manager_b = CourseRAGManager(settings_b)
    manager_a.connect_elasticsearch()
    manager_b.connect_elasticsearch()
    
    results = []
    for item in eval_set[:num_queries]:
        res_a = manager_a.search_faq(item['query'], 3, None)
        res_b = manager_b.search_faq(item['query'], 3, None)
        
        results.append({
            'query': item['query'][:60],
            'expected_course': item['course'],
            'config_a_answer': res_a[0]['_source']['question'][:80] if res_a else 'NONE',
            'config_a_course': res_a[0]['_source']['course'] if res_a else 'NONE',
            'config_a_score': round(res_a[0]['_score'], 2) if res_a else 0,
            'config_b_answer': res_b[0]['_source']['question'][:80] if res_b else 'NONE',
            'config_b_course': res_b[0]['_source']['course'] if res_b else 'NONE',
            'config_b_score': round(res_b[0]['_score'], 2) if res_b else 0,
        })
    
    df = pd.DataFrame(results)
    
    # Calculate winners
    a_wins = 0
    b_wins = 0
    ties = 0
    
    for _, row in df.iterrows():
        if row['config_a_score'] > row['config_b_score']:
            a_wins += 1
        elif row['config_b_score'] > row['config_a_score']:
            b_wins += 1
        else:
            ties += 1
    
    print("=" * 80)
    print(f"A/B TEST: {config_a} vs {config_b}")
    print("=" * 80)
    
    for _, row in df.iterrows():
        print(f"\nQ: {row['query']}")
        print(f"Expected: {row['expected_course']}")
        print(f"[A] {config_a}: {row['config_a_course']} (score: {row['config_a_score']})")
        print(f"    {row['config_a_answer'][:60]}...")
        print(f"[B] {config_b}: {row['config_b_course']} (score: {row['config_b_score']})")
        print(f"    {row['config_b_answer'][:60]}...")
        print("-" * 40)
    
    print("\n" + "=" * 80)
    print("WINNER SUMMARY (by score only)")
    print("=" * 80)
    print(f"Config A ({config_a}) wins: {a_wins}")
    print(f"Config B ({config_b}) wins: {b_wins}")
    print(f"Ties: {ties}")
    
    if a_wins > b_wins:
        print(f"\n🏆 WINNER: {config_a}")
    elif b_wins > a_wins:
        print(f"\n🏆 WINNER: {config_b}")
    else:
        print(f"\n🤝 TIE")
    
    return df

def calculate_winners_from_df(df, config_a_name, config_b_name):
    a_wins = 0
    b_wins = 0
    ties = 0
    
    for _, row in df.iterrows():
        if row['config_a_score'] > row['config_b_score']:
            a_wins += 1
        elif row['config_b_score'] > row['config_a_score']:
            b_wins += 1
        else:
            ties += 1
    
    print("=" * 40)
    print("WINNER SUMMARY (by score only)")
    print("=" * 40)
    print(f"Config A ({config_a_name}) wins: {a_wins}")
    print(f"Config B ({config_b_name}) wins: {b_wins}")
    print(f"Ties: {ties}")
    
    if a_wins > b_wins:
        print(f"\n🏆 WINNER: {config_a_name}")
    elif b_wins > a_wins:
        print(f"\n🏆 WINNER: {config_b_name}")
    else:
        print(f"\n🤝 TIE")
    
    return {'A': a_wins, 'B': b_wins, 'TIE': ties}

if __name__ == "__main__":
    df = run_ab_test("baseline_bm25", "global_cross_fields", num_queries=10)