# /home/admin/LLM/LLM/01/web/test_llm_judge.py

import sys
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')

from dotenv import load_dotenv
load_dotenv('/home/admin/LLM/LLM/01/web/configs/.env')

from src.search import CourseRAGManager
from src.config_manager import load_full_config

config = load_full_config('bm25_default')
manager = CourseRAGManager(config)
manager.connect_elasticsearch()

test_queries = [
    "How to fix docker container not starting",
    "What is the best way to learn Python for data science",
    "How to deploy model using MLflow"
]

print("=== LLM AS JUDGE TEST ===\n")

for i, query in enumerate(test_queries, 1):
    print(f"Query {i}: {query}")
    print("-" * 50)
    
    hits = manager.search_faq(query, 3, None)
    
    if hits:
        contexts = [hit['_source'].get('text', '') for hit in hits]
        response_text = contexts[0] if contexts else ""
        
        print(f"Top result: {hits[0]['_source'].get('question', 'N/A')}")
        print(f"Course: {hits[0]['_source'].get('course', 'N/A')}")
        print(f"Score: {hits[0]['_score']:.2f}")
        
        print("\nEvaluating with LLM...")
        try:
            triad_result = manager.evaluate_rag_triad(query, response_text, contexts)
            
            print(f"Faithful: {triad_result.get('faithful')}")
            print(f"Faithfulness Score: {triad_result.get('faithfulness_score')}")
            print(f"Relevant: {triad_result.get('relevant')}")
            print(f"Relevancy Score: {triad_result.get('relevancy_score')}")
            print(f"Context Utilization: {triad_result.get('context_utilization_rate', 0):.1%}")
            print(f"Contexts Used: {triad_result.get('contexts_used')}/{triad_result.get('contexts_provided')}")
        except Exception as e:
            print(f"Evaluation failed: {e}")
    
    print("\n" + "=" * 50 + "\n")

print("Test complete")
