# /home/admin/LLM/LLM/01/web/eval/llm_judge_with_examples.py

import sys
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')

import json
import random
import litellm
from dotenv import load_dotenv
load_dotenv('/home/admin/LLM/LLM/01/web/configs/.env')

def evaluate_single_query(query, context, answer):
    prompt = f"""You are a RAG evaluation expert. Evaluate if the ANSWER is:
1. Faithful: Does the answer come ONLY from the CONTEXT? (no hallucination)
2. Relevant: Does the answer directly address the QUESTION?

Examples:
QUESTION: "How to fix Docker error?"
CONTEXT: "Docker errors can be fixed by restarting the daemon."
ANSWER: "Restart the Docker daemon."
RESULT: {{"faithful": true, "relevant": true}}

QUESTION: "How to install Python?"
CONTEXT: "Python can be installed via apt or brew."
ANSWER: "Use pip to install packages."
RESULT: {{"faithful": false, "relevant": false}}

QUESTION: "What is the staging dataset?"
CONTEXT: "A staging dataset is used for data preparation before production."
ANSWER: "A staging dataset is used to validate data quality before moving to production."
RESULT: {{"faithful": true, "relevant": true}}

Now evaluate:

QUESTION: {query}

CONTEXT: {context[:500]}

ANSWER: {answer[:300]}

Return ONLY valid JSON: {{"faithful": true/false, "relevant": true/false}}"""

    try:
        response = litellm.completion(
            model="nvidia_nim/nvidia/nemotron-mini-4b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=100
        )
        result_text = response.choices[0].message.content.strip()
        result = json.loads(result_text)
        return {
            'faithful': result.get('faithful', False),
            'relevant': result.get('relevant', False)
        }
    except Exception as e:
        print(f"Error: {e}")
        print(f"Response: {result_text if 'result_text' in locals() else 'None'}")
        return {'faithful': False, 'relevant': False}

def evaluate_config(config_name, samples, num_queries=10):
    print(f"\n{'='*50}")
    print(f"Evaluating: {config_name}")
    print('='*50)
    
    with open(f'/home/admin/LLM/LLM/01/web/experiments/results/{config_name}.json', 'r') as f:
        data = json.load(f)
    
    results_by_query = {}
    for r in data['results']:
        if r['k'] == 5:
            results_by_query[r['query']] = {
                'success': r['success'],
                'contexts': r.get('contexts', []),
            }
    
    faithful_count = 0
    relevant_count = 0
    
    for i, query in enumerate(samples[:num_queries], 1):
        if query not in results_by_query:
            continue
        
        contexts = results_by_query[query]['contexts']
        if not contexts:
            continue
        
        top_context = contexts[0]
        answer = top_context[:300]
        
        print(f"\n[{i}/{num_queries}] {query[:60]}...")
        
        result = evaluate_single_query(query, top_context, answer)
        
        print(f"  Faithful: {result['faithful']}")
        print(f"  Relevant: {result['relevant']}")
        
        if result['faithful']:
            faithful_count += 1
        if result['relevant']:
            relevant_count += 1
    
    print(f"\n{config_name} Summary:")
    print(f"  Faithful answers: {faithful_count}/{num_queries} ({faithful_count/num_queries*100:.1f}%)")
    print(f"  Relevant answers: {relevant_count}/{num_queries} ({relevant_count/num_queries*100:.1f}%)")

if __name__ == "__main__":
    with open('/home/admin/LLM/LLM/01/web/experiments/results/bm25_default.json', 'r') as f:
        data = json.load(f)
    all_queries = list(set(r['query'] for r in data['results'] if r['k'] == 5))
    samples = random.sample(all_queries, min(10, len(all_queries)))
    
    print(f"Testing {len(samples)} random queries with examples\n")
    
    for config in ['bm25_default']:
        evaluate_config(config, samples, num_queries=10)
