import os
import json
from dotenv import load_dotenv
from search import CourseRAGManager
from core import build_prompt, query_llm

def load_settings(filename="settings.json"):
    with open(filename, 'r') as f:
        return json.load(f)

def main():
    # 1. Setup
    load_dotenv()
    settings = load_settings()
    nv_key = os.getenv("NVIDIA_API_KEY")
    or_key = os.getenv("OPENROUTER_API_KEY")
    
    # 2. Retrieval
    rag = CourseRAGManager(settings)
    rag.connect_elasticsearch(settings.get("es_host", "http://localhost:9200"))
    
    question = "How do I run docker?"
    records = rag.search_faq(question)
    
    # 3. Generation
    prompt = build_prompt(question, records)
    answer, source = query_llm(prompt, nv_key, or_key, settings)
    
    print(f"\n[{source}] Answer: {answer}")

if __name__ == "__main__":
    main()
