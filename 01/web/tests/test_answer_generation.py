import sys
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')

from src.search import CourseRAGManager
from src.config_manager import load_config

print("=" * 60)
print("TESTING LLAMAINDEX ANSWER GENERATION")
print("=" * 60)

# Load config with query engine enabled
settings = load_config('../experiments/configs/llama_full.json')
settings['use_llama_query'] = True  # Enable query engine
settings['build_llama_index'] = False  # Already built

manager = CourseRAGManager(settings)
manager.connect_elasticsearch()

query = "When does the course start?"
print(f"\n📝 Query: {query}")

results = manager.search_faq(query, 1, 'data-engineering-zoomcamp')

if results:
    print(f"\n📄 Retrieved Document:")
    print(f"   Course: {results[0]['_source']['course']}")
    print(f"   Question: {results[0]['_source']['question']}")
    print(f"   Answer: {results[0]['_source']['text'][:200]}...")
    
    print(f"\n🤖 LLM Generated Answer:")
    print(f"   {results[0]['_source']['text'][:300]}...")
else:
    print("No results found")
