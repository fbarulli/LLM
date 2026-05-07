# /home/admin/LLM/LLM/01/web/test_rag_triad.py

import sys
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')

from dotenv import load_dotenv
load_dotenv('/home/admin/LLM/LLM/01/web/configs/.env')

import json
from src.search import CourseRAGManager
from src.config_manager import load_full_config

config = load_full_config('bm25_default')
manager = CourseRAGManager(config)
manager.connect_elasticsearch()

test_query = "How to fix docker container not starting"

print(f"Query: {test_query}")
print("-" * 50)

hits = manager.search_faq(test_query, 3, None)

if hits:
    contexts = [hit['_source'].get('text', '') for hit in hits]
    response_text = contexts[0] if contexts else ""
    
    print(f"Top result: {hits[0]['_source'].get('question', 'N/A')}")
    print(f"Course: {hits[0]['_source'].get('course', 'N/A')}")
    print(f"Score: {hits[0]['_score']:.2f}")
    print("-" * 50)
    
    triad_result = manager.evaluate_rag_triad(test_query, response_text, contexts)
    
    print("\n=== RAG TRIAD RESULTS ===")
    print(f"Faithful: {triad_result.get('faithful')}")
    print(f"Faithfulness Score: {triad_result.get('faithfulness_score')}")
    print(f"Relevant: {triad_result.get('relevant')}")
    print(f"Relevancy Score: {triad_result.get('relevancy_score')}")
    print(f"Context Utilization Rate: {triad_result.get('context_utilization_rate', 0):.1%}")
    print(f"Contexts Used: {triad_result.get('contexts_used')}/{triad_result.get('contexts_provided')}")
else:
    print("No results found")
