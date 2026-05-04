# /home/admin/LLM/LLM/01/web/scripts/generate_hard_eval.py

import sys
import os
import json
import time
from typing import List, Dict

# Add web root to path
web_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, web_root)

from src.prompt_manager import query_llm
from src.core import generate_document_id
from src.logger_config import logger

def load_settings():
    with open(os.path.join(web_root, "settings.json"), "r") as f:
        return json.load(f)

def load_documents():
    with open(os.path.join(web_root, "documents.json"), "r") as f:
        return json.load(f)

def generate_paraphrases(original_question: str, settings: dict, num_variations: int = 3) -> List[str]:
    prompt = f"""Generate {num_variations} alternative ways a user might ask this question. Use natural, conversational language. Change wording and sentence structure. Do NOT use the original wording. Keep each under 15 words. Output one per line, numbered.

Original: {original_question}

Variations:"""

    response, provider = query_llm(prompt, settings, {"task": "paraphrase"})
    
    if not response or "Error" in response:
        logger.warning(f"Failed for: {original_question[:50]}...")
        return []
    
    # Parse numbered lines
    lines = response.strip().split('\n')
    paraphrases = []
    for line in lines:
        cleaned = line.strip()
        if cleaned and cleaned[0].isdigit():
            cleaned = cleaned.split('.', 1)[-1].strip()
            cleaned = cleaned.split(')', 1)[-1].strip()
        if cleaned and len(cleaned) > 5 and len(cleaned) < 100:
            paraphrases.append(cleaned)
    
    return paraphrases[:num_variations]

def main():
    print("=" * 60)
    print("GENERATING HARD EVALUATION SET")
    print("=" * 60)
    
    settings = load_settings()
    data = load_documents()
    
    # Use OpenRouter with rate limit delay
    questions_per_course = 10
    delay_seconds = 4  # OpenRouter free tier: 16 requests/min → 3.75s between, use 4s
    
    print(f"\n🤖 Provider: OpenRouter (rate limit: 1 request every {delay_seconds}s)")
    print(f"📊 Questions per course: {questions_per_course}")
    print(f"⏱️  Estimated time: ~{questions_per_course * len(data) * delay_seconds / 60:.1f} minutes\n")
    
    hard_eval = []
    
    for course in data:
        course_name = course["course"]
        print(f"\n📚 Processing: {course_name}")
        
        for i, doc in enumerate(course["documents"][:questions_per_course]):
            original = doc["question"]
            # Clean prefix if present
            if " - " in original:
                original = original.split(" - ", 1)[1].strip()
            
            print(f"  Q{i+1}: {original[:50]}...", end=" ", flush=True)
            
            paraphrases = generate_paraphrases(original, settings, num_variations=3)
            
            if paraphrases:
                clean_doc = {
                    "text": doc["text"],
                    "question": original,
                    "course": course_name
                }
                expected_id = generate_document_id(clean_doc)
                
                hard_eval.append({
                    "original_question": original,
                    "expected_id": expected_id,
                    "expected_course": course_name,
                    "expected_text": doc["text"][:200],
                    "paraphrased_queries": paraphrases
                })
                print(f"✓ Generated {len(paraphrases)} variants")
            else:
                print(f"✗ Failed")
            
            time.sleep(delay_seconds)
    
    # Save to file
    output_path = os.path.join(web_root, "experiments", "hard_eval_set.json")
    with open(output_path, "w") as f:
        json.dump(hard_eval, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"✅ Saved {len(hard_eval)} hard queries to {output_path}")
    print("=" * 60)
    
    # Preview
    print("\n📋 PREVIEW (first 3):")
    for item in hard_eval[:3]:
        print(f"\nOriginal: {item['original_question']}")
        print(f"Variants: {item['paraphrased_queries']}")

if __name__ == "__main__":
    main()