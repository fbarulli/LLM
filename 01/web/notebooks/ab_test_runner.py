# /home/admin/LLM/LLM/01/web/notebooks/ab_test_runner.py

from src.search import CourseRAGManager
from src.config_manager import load_config
from src.run_stats import get_eval_set

def run_ab_test(config_a, config_b, num_queries=10, k=3):
    eval_set = get_eval_set("documents.json", n_per_course=num_queries // 3)
    
    manager_a = CourseRAGManager(load_config(f"experiments/configs/{config_a}.json"))
    manager_b = CourseRAGManager(load_config(f"experiments/configs/{config_b}.json"))
    manager_a.connect_elasticsearch()
    manager_b.connect_elasticsearch()
    
    results = []
    for item in eval_set[:num_queries]:
        res_a = manager_a.search_faq(item['query'], k, None)
        res_b = manager_b.search_faq(item['query'], k, None)
        
        results.append({
            'query': item['query'][:60],
            'expected_course': item['course'],
            'a_answer': res_a[0]['_source']['question'][:80] if res_a else 'NONE',
            'a_course': res_a[0]['_source']['course'] if res_a else 'NONE',
            'a_score': round(res_a[0]['_score'], 2) if res_a else 0,
            'b_answer': res_b[0]['_source']['question'][:80] if res_b else 'NONE',
            'b_course': res_b[0]['_source']['course'] if res_b else 'NONE',
            'b_score': round(res_b[0]['_score'], 2) if res_b else 0,
        })
    
    return pd.DataFrame(results)

# Run test
ab_df = run_ab_test("baseline_bm25", "global_cross_fields", num_queries=10)
display(ab_df)