import sys
import os

# Add web root to path
web_root = '/home/admin/LLM/LLM/01/web'
sys.path.insert(0, web_root)

print("[DEBUG] Starting imports...")
from src.search import CourseRAGManager
from src.config_manager import load_config
print("[DEBUG] Imports successful")

# Change to web root for relative paths
os.chdir(web_root)
print(f"[DEBUG] Changed to directory: {os.getcwd()}")

# Load config
print("[DEBUG] Loading config...")
settings = load_config('experiments/configs/llama_full.json')
print(f"[DEBUG] Config loaded: use_evaluators={settings.get('use_evaluators', False)}")

# Initialize manager
print("[DEBUG] Initializing manager...")
manager = CourseRAGManager(settings)
manager.connect_elasticsearch()
print("[DEBUG] Manager initialized")

# Get a search result
query = 'When does the course start?'
print(f"\n[DEBUG] Running search for: {query}")
results = manager.search_faq(query, 3, None)

if results:
    response_text = results[0]['_source']['text']
    context = [results[0]['_source']['text']]
    
    print('=' * 60)
    print('TESTING EVALUATORS')
    print('=' * 60)
    print(f'\n📝 Query: {query}')
    print(f'\n📄 Response (first 200 chars): {response_text[:200]}...')
    print(f'\n📚 Contexts count: {len(context)}')
    print(f'📚 Context (first 200 chars): {context[0][:200]}...')
    
    # Run evaluation
    print('\n[DEBUG] Calling evaluate_response...')
    evaluation = manager.evaluate_response(query, response_text, context)
    
    print('\n' + '=' * 60)
    print('📊 EVALUATION RESULTS')
    print('=' * 60)
    print(f'   Faithful: {evaluation.get("faithful", "N/A")}')
    print(f'   Faithfulness Score: {evaluation.get("faithfulness_score", "N/A")}')
    print(f'   Relevant: {evaluation.get("relevant", "N/A")}')
    print(f'   Relevancy Score: {evaluation.get("relevancy_score", "N/A")}')
    print(f'   Error: {evaluation.get("error", "None")}')
else:
    print("No results found")

print("\n[DEBUG] Test complete")
