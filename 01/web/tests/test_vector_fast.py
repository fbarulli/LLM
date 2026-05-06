import sys
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')

from src.search import CourseRAGManager
from src.config_manager import load_config
import json

print("Testing VECTOR search with 5 queries...")

settings = load_config('../settings.json')
manager = CourseRAGManager(settings)
manager.connect_elasticsearch()

# Load hard eval set
with open('../experiments/hard_eval_set.json', 'r') as f:
    hard_eval = json.load(f)

# Take 5 queries
test_queries = []
for item in hard_eval[:2]:
    for variant in item['paraphrased_queries'][:1]:
        test_queries.append({
            'query': variant,
            'expected_course': item['expected_course']
        })

print(f"\nTesting {len(test_queries)} queries:\n")

for q in test_queries:
    print(f"Query: {q['query']}")
    results = manager.search_faq(q['query'], 3, q['expected_course'])
    if results:
        print(f"  → {results[0]['_source']['course']}: {results[0]['_source']['question'][:60]}...")
    else:
        print("  → No results")
    print()

print("✅ Vector search is working")
