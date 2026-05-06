# /home/admin/LLM/LLM/01/web/scripts/test_hard_eval.py

import sys
import os

# Add web root to path
web_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, web_root)

import json
from src.prompt_manager import query_llm
from src.core import generate_document_id

def test_single_question():
    print("=" * 60)
    print("TESTING SINGLE QUESTION PARAPHRASE")
    print("=" * 60)
    
    # Load settings
    with open(os.path.join(web_root, "settings.json"), "r") as f:
        settings = json.load(f)
    
    # Load first question from documents
    with open(os.path.join(web_root, "documents.json"), "r") as f:
        data = json.load(f)
    
    # Get first question from data-engineering-zoomcamp
    course = data[0]
    doc = course["documents"][0]
    original = doc["question"]
    
    # Clean prefix if present
    if " - " in original:
        original = original.split(" - ", 1)[1].strip()
    
    print(f"\n📝 Original question: {original}")
    print(f"   Course: {course['course']}")
    
    # Generate paraphrases
    prompt = f"""Generate 3 alternative ways a user might ask this question. Use natural, conversational language. Change wording and sentence structure. Do NOT use the original wording. Keep each under 15 words. Output one per line, numbered.

Original: {original}

Variations:"""
    
    print(f"\n🤖 Sending to LLM...")
    response, provider = query_llm(prompt, settings, {"task": "test"})
    
    print(f"   Provider: {provider}")
    print(f"\n📤 Raw response:\n{response}")
    
    # Parse response
    if response and "Error" not in response:
        lines = response.strip().split('\n')
        paraphrases = []
        for line in lines:
            cleaned = line.strip()
            if cleaned and cleaned[0].isdigit():
                cleaned = cleaned.split('.', 1)[-1].strip()
                cleaned = cleaned.split(')', 1)[-1].strip()
            if cleaned and len(cleaned) > 5:
                paraphrases.append(cleaned)
        
        print(f"\n✅ Parsed variations ({len(paraphrases)}):")
        for i, p in enumerate(paraphrases, 1):
            print(f"   {i}. {p}")
        
        # Get expected ID
        clean_doc = {
            "text": doc["text"],
            "question": original,
            "course": course["course"]
        }
        expected_id = generate_document_id(clean_doc)
        print(f"\n📌 Expected document ID: {expected_id}")
        
    else:
        print("\n❌ Failed to generate variations")

if __name__ == "__main__":
    test_single_question()