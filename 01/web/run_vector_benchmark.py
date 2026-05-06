import sys
import json
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')

from src.search import CourseRAGManager
from src.config_manager import load_config

def test_config(config_name, test_queries):
    print(f"\n🧪 Testing: {config_name}")
    settings = load_config(f"experiments/configs/{config_name}.json")
    manager = CourseRAGManager(settings)
    manager.connect_elasticsearch()
    
    correct = 0
    for q in test_queries:
        results = manager.search_faq(q['query'], 5, q['expected_course'])
        if q['expected_id'] in [hit['_id'] for hit in results]:
            correct += 1
    return correct

# Load hard eval set
with open('experiments/hard_eval_set.json', 'r') as f:
    hard_eval = json.load(f)

# Flatten for testing
test_queries = []
for item in hard_eval[:10]:  # 10 original questions
    for variant in item['paraphrased_queries']:
        test_queries.append({
            'query': variant,
            'expected_id': item['expected_id'],
            'expected_course': item['expected_course']
        })

print(f"Testing {len(test_queries)} queries per config")

# Run each config (separate process to avoid model state issues)
for config in ['baseline_bm25', 'vector_bm25', 'hybrid_bm25']:
    correct = test_config(config, test_queries)
    print(f"  {config}: {correct}/{len(test_queries)} = {correct/len(test_queries):.1%}")
