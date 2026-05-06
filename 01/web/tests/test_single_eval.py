import sys
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')
import json
from src.search import CourseRAGManager
from src.config_manager import load_config

print("Testing single evaluation...")

# Load one result
with open('../experiments/results/baseline_bm25_hard_full.json', 'r') as f:
    data = json.load(f)

# Find first result to evaluate
result = None
for r in data['results']:
    if r['k'] == 5 and r['found_id'] != 'NONE':
        result = r
        break

if not result:
    print("No results found")
    sys.exit(1)

print(f"Query: {result['query'][:60]}...")
print(f"Response length: {len(result.get('best_text', ''))}")

# Initialize evaluator
settings = load_config('../experiments/configs/llama_full.json')
manager = CourseRAGManager(settings)
manager.connect_elasticsearch()

# Single evaluation
response = result.get('best_text', '')
query = result['query']
context = [response]

print("Calling evaluate_response (this may take 5-10 seconds)...")
eval_result = manager.evaluate_response(query, response, context)

print(f"\nResults:")
print(f"  Faithful: {eval_result.get('faithful')}")
print(f"  Relevant: {eval_result.get('relevant')}")
print(f"  Error: {eval_result.get('error', 'None')}")
