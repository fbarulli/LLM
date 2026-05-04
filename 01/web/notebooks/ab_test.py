# /home/admin/LLM/LLM/01/web/notebooks/ab_test.py

import sys
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')

from src.search import CourseRAGManager
from src.config_manager import load_config
from src.run_stats import get_eval_set
import pandas as pd
from datetime import datetime

def ab_test(config_a_name, config_b_name, num_queries=20, k=3):
    """
    A/B test between two search configs.
    Shows results side by side for human evaluation.
    """
    print("=" * 80)
    print(f"A/B TEST: {config_a_name} vs {config_b_name}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Load configs
    settings_a = load_config(f"experiments/configs/{config_a_name}.json")
    settings_b = load_config(f"experiments/configs/{config_b_name}.json")
    
    manager_a = CourseRAGManager(settings_a)
    manager_b = CourseRAGManager(settings_b)
    manager_a.connect_elasticsearch()
    manager_b.connect_elasticsearch()
    
    # Load eval set
    eval_set = get_eval_set("documents.json", n_per_course=num_queries // 3)
    
    results = []
    
    for i, item in enumerate(eval_set[:num_queries]):
        query = item['query']
        expected_course = item['course']
        
        # Run both configs
        results_a = manager_a.search_faq(query, override_size=k, course_context=None)
        results_b = manager_b.search_faq(query, override_size=k, course_context=None)
        
        # Extract top result info
        top_a = results_a[0] if results_a else None
        top_b = results_b[0] if results_b else None
        
        results.append({
            'query_id': i + 1,
            'query': query,
            'expected_course': expected_course,
            'config_a_course': top_a['_source']['course'] if top_a else 'NONE',
            'config_a_question': top_a['_source']['question'][:80] if top_a else 'NONE',
            'config_a_score': round(top_a['_score'], 2) if top_a else 0,
            'config_b_course': top_b['_source']['course'] if top_b else 'NONE',
            'config_b_question': top_b['_source']['question'][:80] if top_b else 'NONE',
            'config_b_score': round(top_b['_score'], 2) if top_b else 0,
            'winner': None  # To be filled by human/LLM
        })
    
    # Display results for human judgment
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 80)
    print("QUERY RESULTS - CHOOSE WINNER PER ROW")
    print("=" * 80)
    
    for _, row in df.iterrows():
        print(f"\n--- Query {row['query_id']}: {row['query']} ---")
        print(f"Expected Course: {row['expected_course']}")
        print(f"\n[A] {config_a_name}:")
        print(f"    Course: {row['config_a_course']}")
        print(f"    Answer: {row['config_a_question']}")
        print(f"    Score: {row['config_a_score']}")
        print(f"\n[B] {config_b_name}:")
        print(f"    Course: {row['config_b_course']}")
        print(f"    Answer: {row['config_b_question']}")
        print(f"    Score: {row['config_b_score']}")
    
    print("\n" + "=" * 80)
    print("TO COLLECT RESULTS:")
    print("Create a DataFrame and mark winners:")
    print("df['winner'] = ['A', 'B', 'A', ...]  # per query")
    print("Then run: df['winner'].value_counts()")
    print("=" * 80)
    
    return df

def calculate_ab_test_results(df):
    """Calculate which config won more queries"""
    if 'winner' not in df.columns:
        print("No winners marked yet. Add 'winner' column with 'A' or 'B' per row")
        return None
    
    wins = df['winner'].value_counts()
    total = len(df)
    
    print("\n=== A/B TEST RESULTS ===")
    print(f"Total queries compared: {total}")
    print(f"Config A wins: {wins.get('A', 0)} ({wins.get('A', 0)/total*100:.1f}%)")
    print(f"Config B wins: {wins.get('B', 0)} ({wins.get('B', 0)/total*100:.1f}%)")
    print(f"Ties: {wins.get('TIE', 0)} ({wins.get('TIE', 0)/total*100:.1f}%)")
    
    return wins

# Run A/B test between baseline and global_cross_fields
df = ab_test("baseline_bm25", "global_cross_fields", num_queries=10, k=3)

# After you manually mark winners, run:
# calculate_ab_test_results(df)