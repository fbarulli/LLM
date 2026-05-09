"""
eval/generation/generate_test_queries.py
=========================================
Generates test queries in batches of 5 Q&A pairs per LLM call.
Shows model diverse examples from different courses/sections.

Output: experiments/eval_queries.json

Run:    uv run python eval/generation/generate_test_queries.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import re
import json
import asyncio
import random
from difflib import SequenceMatcher
from collections import defaultdict
from dotenv import load_dotenv
from litellm import acompletion
from qdrant_client import QdrantClient

load_dotenv('configs/.env')
os.environ["NVIDIA_NIM_API_KEY"] = os.getenv("NVIDIA_API_KEY")

from eval.judges.shared import JUDGE_MODEL_70B, run_sequential

OUTPUT = 'experiments/eval_queries.json'
PROMPTS_FILE = 'eval/generation/prompts.json'
BATCH_SIZE = 5
TOTAL_DOCS = 60  # 12 batches × 5 docs
MODEL = JUDGE_MODEL_70B


def load_prompts() -> dict:
    with open(PROMPTS_FILE) as f:
        return json.load(f)


def get_diverse_sample(client, n=TOTAL_DOCS) -> list:
    """Get documents ensuring diversity across courses and sections."""
    # Get all docs
    all_docs = []
    offset = 0
    while True:
        results = client.scroll(
            collection_name='faqs', limit=100, offset=offset, with_payload=True
        )
        points, next_offset = results
        if not points:
            break
        for point in points:
            all_docs.append(point.payload)
        offset = next_offset

    # Group by (course, section)
    from itertools import groupby
    all_docs.sort(key=lambda d: (d['course'], d.get('section', '')))
    
    groups = defaultdict(list)
    for doc in all_docs:
        groups[(doc['course'], doc.get('section', ''))].append(doc)

    # Sample evenly from each group
    random.seed(42)
    sampled = []
    group_list = list(groups.values())
    random.shuffle(group_list)
    
    for group in group_list:
        if len(sampled) >= n:
            break
        if group:
            sampled.append(random.choice(group))

    random.shuffle(sampled)
    return sampled[:n]


def build_qa_pairs(docs: list) -> str:
    """Format 5 Q&A pairs for the prompt."""
    parts = []
    for i, doc in enumerate(docs, 1):
        parts.append(
            f"FAQ {i}:\n"
            f"Course: {doc['course']}\n"
            f"Question: {doc['question']}\n"
            f"Answer: {doc['answer'][:400]}\n"
        )
    return "\n".join(parts)


async def generate_batch(docs: list, prompt_name: str, template: str) -> list:
    """Generate queries for a batch of 5 docs using one prompt strategy."""
    qa_pairs = build_qa_pairs(docs)
    prompt = template.format(qa_pairs=qa_pairs)
    
    try:
        response = await acompletion(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r'```(?:json)?|```', '', raw).strip()
        result = json.loads(raw)

        if not isinstance(result, dict):
            raise ValueError(f"Expected dict, got {type(result)}")

        # Parse results back to per-document format
        output = []
        for i, doc in enumerate(docs, 1):
            variations = result.get(str(i), [])
            if not isinstance(variations, list):
                variations = []
            output.append({
                'original_question': doc['question'],
                'expected_id': doc['es_id'],
                'course': doc['course'],
                'section': doc.get('section', ''),
                'prompt_strategy': prompt_name,
                'variations': variations[:3],
            })
        return output

    except Exception as e:
        print(f"  [{prompt_name}] batch failed: {e}")
        return [
            {
                'original_question': doc['question'],
                'expected_id': doc['es_id'],
                'course': doc['course'],
                'section': doc.get('section', ''),
                'prompt_strategy': prompt_name,
                'variations': [
                    f"help with {doc['question'][:30]}",
                    doc['question'][:40],
                    f"how to {doc['question'][:30]}",
                ],
            }
            for doc in docs
        ]


async def main():
    prompts = load_prompts()
    client = QdrantClient('localhost', port=6333)
    docs = get_diverse_sample(client, TOTAL_DOCS)
    
    print(f"Documents: {len(docs)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Prompts: {list(prompts.keys())}")
    print(f"Total LLM calls: {len(prompts) * (len(docs) // BATCH_SIZE)}\n")

    all_results = []
    
    # Process in batches, each batch gets all prompts
    for batch_num in range(0, len(docs), BATCH_SIZE):
        batch = docs[batch_num:batch_num + BATCH_SIZE]
        if len(batch) < BATCH_SIZE:
            break
        
        print(f"Batch {batch_num // BATCH_SIZE + 1}/{(len(docs) // BATCH_SIZE)}")
        print(f"  Courses: {set(d['course'] for d in batch)}")
        
        for prompt_name, info in prompts.items():
            results = await generate_batch(batch, prompt_name, info['template'])
            all_results.extend(results)
            count = sum(len(r['variations']) for r in results)
            print(f"    [{prompt_name}] {count} queries generated")
        
        # Gap between batches
        await asyncio.sleep(3)

    # Group by document (one doc may appear in multiple batches with different prompts)
    by_doc = defaultdict(lambda: {'prompt_results': {}})
    for r in all_results:
        key = r['original_question']
        by_doc[key]['original_question'] = r['original_question']
        by_doc[key]['expected_id'] = r['expected_id']
        by_doc[key]['course'] = r['course']
        by_doc[key]['section'] = r['section']
        by_doc[key]['prompt_results'][r['prompt_strategy']] = r['variations']

    queries = list(by_doc.values())
    total = sum(len(v) for doc in queries for v in doc['prompt_results'].values())

    output = {
        'metadata': {
            'description': 'Batch-generated test queries with 5 Q&A pairs per call',
            'model': MODEL,
            'total_documents': len(queries),
            'total_queries': total,
            'batch_size': BATCH_SIZE,
            'prompt_strategies': list(prompts.keys()),
        },
        'queries': queries,
    }

    os.makedirs('experiments', exist_ok=True)
    with open(OUTPUT, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nTotal: {len(queries)} docs, {total} queries")
    print(f"Saved → {OUTPUT}")


if __name__ == '__main__':
    asyncio.run(main())
