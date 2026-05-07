# /home/admin/LLM/LLM/01/web/eval/llm_judge_debug.py

import sys
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')

import json
import litellm
from dotenv import load_dotenv
load_dotenv('/home/admin/LLM/LLM/01/web/configs/.env')

def test_single_evaluation():
    query = "Why do we need the Staging dataset?"
    context = "A staging dataset is used to prepare and validate data before production."
    answer = "A staging dataset is used to prepare and validate data before production."
    
    prompt = f"""Evaluate if the ANSWER is faithful to the CONTEXT and relevant to the QUESTION.

QUESTION: {query}

CONTEXT: {context}

ANSWER: {answer}

Return ONLY valid JSON: {{"faithful": true/false, "relevant": true/false}}"""

    print("PROMPT:")
    print(prompt)
    print("\n" + "="*50)
    
    response = litellm.completion(
        model="nvidia_nim/nvidia/nemotron-mini-4b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=100
    )
    
    result_text = response.choices[0].message.content
    print("RAW RESPONSE:")
    print(repr(result_text))
    print("\n" + "="*50)
    
    print("RESPONSE CONTENT:")
    print(result_text)

if __name__ == "__main__":
    test_single_evaluation()
