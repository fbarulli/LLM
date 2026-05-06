import sys
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')
import json
import pandas as pd
from src.search import CourseRAGManager
from src.config_manager import load_config

def inspect_config(config_name, config_path, test_queries, num_examples=5):
    print(f"\n{'='*60}")
    print(f"INSPECTING: {config_name}")
    print('='*60)
    
    settings = load_config(config_path)
    manager = CourseRAGManager(settings)
    manager.connect_elasticsearch()
    
    # Evaluate each query
    results = []
    for i, q in enumerate(test_queries):
        search_results = manager.search_faq(q['query'], 5, q['expected_course'])
        if search_results:
            top = search_results[0]
            success = q['expected_id'] == top['_id']
            
            results.append({
                'query': q['query'],
                'expected_course': q['expected_course'],
                'found_course': top['_source']['course'],
                'found_question': top['_source']['question'],
                'found_answer': top['_source']['text'][:200],
                'success': success,
                'score': top['_score']
            })
        else:
            results.append({
                'query': q['query'],
                'expected_course': q['expected_course'],
                'found_course': 'NONE',
                'found_question': 'NONE',
                'found_answer': 'NONE',
                'success': False,
                'score': 0
            })
    
    df = pd.DataFrame(results)
    
    # Summary
    print(f"\n📊 Summary:")
    print(f"   Total: {len(df)}")
    print(f"   Success: {df['success'].sum()} ({df['success'].mean()*100:.1f}%)")
    print(f"   Cross-course: {(df['expected_course'] != df['found_course']).sum()} ({(df['expected_course'] != df['found_course']).mean()*100:.1f}%)")
    
    # Success examples
    print(f"\n✅ SUCCESS EXAMPLES (first {num_examples}):")
    successes = df[df['success'] == True].head(num_examples)
    for _, row in successes.iterrows():
        print(f"\n   Query: {row['query'][:80]}...")
        print(f"   Expected course: {row['expected_course']}")
        print(f"   Found: {row['found_course']} - {row['found_question'][:60]}...")
        print(f"   Answer snippet: {row['found_answer'][:100]}...")
    
    # Failure examples
    print(f"\n❌ FAILURE EXAMPLES (first {num_examples}):")
    failures = df[df['success'] == False].head(num_examples)
    for _, row in failures.iterrows():
        print(f"\n   Query: {row['query'][:80]}...")
        print(f"   Expected course: {row['expected_course']}")
        print(f"   Found: {row['found_course']} - {row['found_question'][:60]}...")
        print(f"   Answer snippet: {row['found_answer'][:100]}...")
    
    # Cross-course examples
    print(f"\n🔄 CROSS-COURSE EXAMPLES (found different course):")
    cross = df[(df['expected_course'] != df['found_course']) & (df['found_course'] != 'NONE')].head(num_examples)
    for _, row in cross.iterrows():
        print(f"\n   Query: {row['query'][:80]}...")
        print(f"   Expected course: {row['expected_course']} → Found: {row['found_course']}")
        print(f"   Found answer: {row['found_answer'][:100]}...")
    
    return df

def main():
    print("=" * 60)
    print("INSPECTING SEARCH RESULTS")
    print("=" * 60)
    
    # Load test queries
    with open('../experiments/merged_fixed_190.json', 'r') as f:
        test_queries = json.load(f)
    
    print(f"📊 Total test queries: {len(test_queries)}")
    
    # Which config to inspect? Change this to 'Vector' or 'Hybrid'
    config_name = "Hybrid"
    config_path = "../experiments/configs/hybrid_bm25.json"
    
    df = inspect_config(config_name, config_path, test_queries, num_examples=5)
    
    # Save to CSV for further analysis
    df.to_csv('../experiments/hybrid_inspection.csv', index=False)
    print(f"\n✅ Saved detailed results to experiments/hybrid_inspection.csv")

if __name__ == "__main__":
    main()
