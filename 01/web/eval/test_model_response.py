# /home/admin/LLM/LLM/01/web/eval/test_model_response.py

import sys
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')

import json
import litellm
from dotenv import load_dotenv
load_dotenv('/home/admin/LLM/LLM/01/web/configs/.env')

# Load a real example
with open('/home/admin/LLM/LLM/01/web/experiments/results/bm25_default.json', 'r') as f:
    data = json.load(f)

# Find first result with context
for r in data['results']:
    if r['k'] == 5 and r.get('contexts') and len(r['contexts']) > 0:
        query = r['query']
        context = r['contexts'][0][:500]
        answer = context[:300]
        break

print(f"Query: {query[:80]}...")
print(f"Context: {context[:100]}...")
print(f"Answer: {answer[:100]}...")
print("\n" + "="*50)

prompt = f"""You are a RAG evaluation expert. Evaluate if the ANSWER is faithful to the CONTEXT and relevant to the QUESTION.

QUESTION: {query}

CONTEXT: {context}

ANSWER: {answer}

Return ONLY valid JSON: {{"faithful": true/false, "relevant": true/false}}"""

response = litellm.completion(
    model="nvidia_nim/nvidia/nemotron-mini-4b-instruct",
    messages=[{"role": "user", "content": prompt}],
    temperature=0,
    max_tokens=100
)

print("MODEL RESPONSE:")
print(response.choices[0].message.content)
