"""
eval/generate_test_queries.py
==============================
Generates varied test queries by taking existing questions and creating
simple variations (word swaps, reorderings) without needing an LLM.

Creates a test set where queries differ from stored questions.

Output: experiments/eval_queries.json

Run:    uv run python eval/generate_test_queries.py
"""
import json
import random
from collections import defaultdict
from elasticsearch import Elasticsearch

OUTPUT = 'experiments/eval_queries.json'
SAMPLE_PER_COURSE = 30

# Simple transformations to create query variations
PREFIXES = [
    "How do I", "I need help with", "Can someone explain",
    "What's the best way to", "I'm stuck on", "Question about",
    "Help with", "Looking for info on", "Trying to understand",
]

SUFFIXES = [
    "?", " - any tips?", " - help!", "??", " - not sure how to proceed",
    " - documentation unclear", "",
]


def create_variations(question: str, num: int = 3) -> list:
    """Create simple variations of a question."""
    variations = []
    clean = question.rstrip('?').strip()
    
    for _ in range(num):
        prefix = random.choice(PREFIXES)
        suffix = random.choice(SUFFIXES)
        
        # Sometimes use the prefix, sometimes just rephrase
        if random.random() < 0.5:
            var = f"{prefix} {clean.lower()}{suffix}"
        else:
            # Just add a suffix or remove words
            words = clean.split()
            if len(words) > 4:
                # Drop some words from middle
                start = words[:2]
                end = words[-2:]
                var = f"{' '.join(start)} ... {' '.join(end)}{suffix}"
            else:
                var = f"{clean}{suffix}"
        
        variations.append(var.strip())
    
    return variations


def main():
    es = Elasticsearch('http://localhost:9200')
    random.seed(42)
    
    # Get all documents
    result = es.search(index='faqs_complete', body={
        'size': 2000,
        'query': {'match_all': {}}
    })
    
    all_docs = [hit['_source'] for hit in result['hits']['hits']]
    
    # Group by course
    by_course = defaultdict(list)
    for doc in all_docs:
        by_course[doc['course']].append(doc)
    
    # Sample from each course
    sampled = []
    for course, docs in by_course.items():
        sample = random.sample(docs, min(SAMPLE_PER_COURSE, len(docs)))
        sampled.extend(sample)
    
    print(f"Sampled {len(sampled)} documents from {len(by_course)} courses")
    
    # Generate variations
    queries = []
    for doc in sampled:
        variations = create_variations(doc['question'], num=3)
        queries.append({
            'original_question': doc['question'],
            'expected_id': doc['id'],
            'course': doc['course'],
            'section': doc['section'],
            'test_queries': variations,
        })
    
    total = sum(len(q['test_queries']) for q in queries)
    
    output = {
        'metadata': {
            'description': 'Test queries for retrieval evaluation - simple variations',
            'total_documents': len(queries),
            'total_queries': total,
        },
        'queries': queries,
    }
    
    with open(OUTPUT, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Generated {total} test queries → {OUTPUT}")


if __name__ == '__main__':
    main()
