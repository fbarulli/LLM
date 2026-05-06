import sys
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')
import json

# Merge eval sets
with open('../experiments/hard_eval_set.json', 'r') as f:
    hard_eval = json.load(f)

with open('../experiments/expanded_eval_set.json', 'r') as f:
    expanded_eval = json.load(f)

# Convert hard_eval to flat format
hard_queries = []
for item in hard_eval:
    for variant in item['paraphrased_queries']:
        hard_queries.append({
            'query': variant,
            'expected_id': item['expected_id'],
            'expected_course': item['expected_course']
        })

print(f"Hard queries: {len(hard_queries)}")
print(f"Expanded queries: {len(expanded_eval)}")
print(f"Total: {len(hard_queries) + len(expanded_eval)}")

# Save merged
merged = hard_queries + expanded_eval
with open('../experiments/merged_190_queries.json', 'w') as f:
    json.dump(merged, f, indent=2)
print(f"✅ Saved merged set to experiments/merged_190_queries.json")
