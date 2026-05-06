import sys
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')
import time
from src.search import CourseRAGManager
from src.config_manager import load_config
import json

print("Loading...")
with open('../experiments/hard_eval_set.json', 'r') as f:
    hard_eval = json.load(f)

# Get first test query
test_query = hard_eval[0]['paraphrased_queries'][0]
expected_course = hard_eval[0]['expected_course']

print(f"Query: {test_query}")
print(f"Course: {expected_course}")

settings = load_config('../experiments/configs/baseline_bm25.json')
manager = CourseRAGManager(settings)
manager.connect_elasticsearch()

print("\nRunning search...")
start = time.time()
search_results = manager.search_faq(test_query, 5, expected_course)
print(f"Search took: {time.time() - start:.2f}s")

if search_results:
    response_text = search_results[0]['_source']['text']
    print(f"Response length: {len(response_text)} chars")
    
    print("\nRunning evaluation (LLM call)...")
    start = time.time()
    evaluation = manager.evaluate_response(test_query, response_text, [response_text])
    print(f"Evaluation took: {time.time() - start:.2f}s")
    
    print(f"\nResult: {evaluation}")
