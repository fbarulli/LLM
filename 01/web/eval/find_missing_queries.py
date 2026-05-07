# /home/admin/LLM/LLM/01/web/eval/find_missing_queries.py

import sys
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')

import json
from eval.eval_set import get_eval_set_from_es

eval_set = get_eval_set_from_es()
all_queries = set(item['original_doc'].get('question', '') for item in eval_set)
print(f"Total queries in ES: {len(all_queries)}")

with open('/home/admin/LLM/LLM/01/web/experiments/results/bm25_default.json', 'r') as f:
    data = json.load(f)

result_queries = set(r['query'] for r in data['results'] if r['k'] == 5)
print(f"Queries in results: {len(result_queries)}")

missing_queries = all_queries - result_queries
print(f"\nMissing queries ({len(missing_queries)}):")

for i, query in enumerate(list(missing_queries)[:10]):
    print(f"{i+1}. {query[:100]}...")

if len(missing_queries) > 10:
    print(f"\n... and {len(missing_queries) - 10} more")
