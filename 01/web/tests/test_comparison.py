import sys
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')
import json

from src.search import CourseRAGManager
from src.config_manager import load_config

print('=' * 60)
print('BM25 vs VECTOR COMPARISON')
print('=' * 60)

try:
    with open('../experiments/hard_eval_set.json', 'r') as f:
        hard_eval = json.load(f)

    test_queries = []
    for item in hard_eval[:3]:
        if item.get('paraphrased_queries'):
            variant = item['paraphrased_queries'][0]
            test_queries.append({
                'query': variant,
                'expected_id': item['expected_id'],
                'expected_course': item['expected_course']
            })

    print(f'\n📊 Testing {len(test_queries)} queries\n')

    # BM25
    print('🧪 Running BM25...')
    settings_bm25 = load_config('../experiments/configs/baseline_bm25.json')
    manager_bm25 = CourseRAGManager(settings_bm25)
    manager_bm25.connect_elasticsearch()

    bm25_correct = 0
    for q in test_queries:
        results = manager_bm25.search_faq(q['query'], 5, q['expected_course'])
        if q['expected_id'] in [hit['_id'] for hit in results]:
            bm25_correct += 1

    # Vector
    print('🧪 Running Vector Search...')
    settings_vec = load_config('../settings.json')
    manager_vec = CourseRAGManager(settings_vec)
    manager_vec.connect_elasticsearch()

    vec_correct = 0
    for q in test_queries:
        results = manager_vec.search_faq(q['query'], 5, None)
        if q['expected_id'] in [hit['_id'] for hit in results]:
            vec_correct += 1

    print('\n📊 RESULTS:')
    print(f'   BM25:   {bm25_correct}/{len(test_queries)} = {bm25_correct/len(test_queries):.1%}')
    print(f'   Vector: {vec_correct}/{len(test_queries)} = {vec_correct/len(test_queries):.1%}')
except Exception as e:
    print(f'Error: {e}')
