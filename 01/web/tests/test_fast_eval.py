import sys
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')
import time
import json
from src.search import CourseRAGManager
from src.config_manager import load_config
from src.fast_evaluator import fast_evaluate

print("Loading...")
with open('../experiments/hard_eval_set.json', 'r') as f:
    hard_eval = json.load(f)

test_query = hard_eval[0]['paraphrased_queries'][0]
expected_course = hard_eval[0]['expected_course']

settings = load_config('../experiments/configs/baseline_bm25.json')
manager = CourseRAGManager(settings)
manager.connect_elasticsearch()

search_results = manager.search_faq(test_query, 5, expected_course)
response_text = search_results[0]['_source']['text']

print(f"Query: {test_query}")
print(f"Response length: {len(response_text)} chars")

print("\nRunning FAST evaluation...")
start = time.time()
result = fast_evaluate(test_query, response_text, response_text)
print(f"Fast evaluation took: {time.time() - start:.2f}s")
print(f"Result: {result}")
