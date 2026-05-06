import sys
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')

from src.search import CourseRAGManager
from src.config_manager import load_config

print("=" * 60)
print("TESTING ANSWER GENERATION - NON EDGE CASE")
print("=" * 60)

settings = load_config('../experiments/configs/llama_full.json')
settings['use_llama_query'] = True
settings['build_llama_index'] = False

manager = CourseRAGManager(settings)
manager.connect_elasticsearch()

# Test a query that requires synthesis, not just exact match
test_queries = [
    ("How do I submit my homework?", "data-engineering-zoomcamp"),
    ("What happens if I submit late?", "machine-learning-zoomcamp"),
    ("Do I get a certificate?", "mlops-zoomcamp"),
]

for query, course in test_queries:
    print(f"\n{'='*50}")
    print(f"📝 Query: \"{query}\"")
    print(f"   Course: {course}")
    
    results = manager.search_faq(query, 2, course)
    
    if results:
        print(f"\n📄 Top Retrieved Document:")
        print(f"   Course: {results[0]['_source']['course']}")
        print(f"   Question: {results[0]['_source']['question'][:80]}...")
        print(f"   Answer snippet: {results[0]['_source']['text'][:150]}...")
        
        print(f"\n🤖 LLM Generated Answer:")
        # Check if using LlamaIndex query engine or raw text
        if settings.get('use_llama_query', False) and manager.query_engine:
            response = manager.query_engine.query(query)
            print(f"   {str(response)[:300]}...")
        else:
            print(f"   {results[0]['_source']['text'][:300]}...")
    else:
        print("   No results found")

print("\n" + "=" * 60)
print("✅ Test complete")
