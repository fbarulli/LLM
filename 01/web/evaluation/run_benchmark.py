#!/usr/bin/env python
"""Run experiment with config overrides via command line."""

import sys
import json
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')

from src.search import CourseRAGManager
from src.core import generate_document_id
from tqdm import tqdm

def load_eval_set():
    with open('documents_datatalks.json', 'r') as f:
        data = json.load(f)
    
    docs = data[0]['documents']
    eval_set = []
    for doc in docs:
        doc_id = generate_document_id({
            "text": doc['text'],
            "question": doc['question'],
            "course": "datatalks-zoomcamp"
        })
        eval_set.append({
            "query": doc['question'],
            "course": "datatalks-zoomcamp",
            "expected_id": doc_id
        })
    return eval_set

def run_benchmark(index_name, search_type, use_vector):
    print(f"\n{'='*50}")
    print(f"Benchmark: {search_type.upper()} on {index_name}")
    print('='*50)
    
    settings = {
        "es_host": "http://localhost:9200",
        "index_name": index_name,
        "search_type": search_type,
        "use_vector": use_vector,
        "embedding_model": "all-MiniLM-L6-v2",
        "boost_question": 20.0,
        "boost_text": 1.0
    }
    
    eval_set = load_eval_set()
    manager = CourseRAGManager(settings)
    manager.connect_elasticsearch()
    
    correct = 0
    for item in tqdm(eval_set, desc="Processing"):
        results = manager.search_faq(item['query'], 5, item['course'])
        if item['expected_id'] in [hit['_id'] for hit in results]:
            correct += 1
    
    recall = correct / len(eval_set) * 100
    print(f"\nRecall@5: {correct}/{len(eval_set)} = {recall:.1f}%")
    return {'type': search_type, 'recall': recall, 'correct': correct, 'total': len(eval_set)}

if __name__ == "__main__":
    import argparse
    
    # Search type configurations
    SEARCH_TYPES = {
        "bm25": {"use_vector": False},
        "vector": {"use_vector": True},
        "hybrid": {"use_vector": True}
    }
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=list(SEARCH_TYPES.keys()), 
                        help="Search type (default: run all)")
    parser.add_argument("--index", default="datatalks-faqs")
    
    args = parser.parse_args()
    
    # Determine which types to run
    if args.type:
        types_to_run = [args.type]
    else:
        types_to_run = list(SEARCH_TYPES.keys())
        print("=" * 60)
        print("RUNNING ALL SEARCH TYPES (BM25, VECTOR, HYBRID)")
        print("=" * 60)
    
    results = []
    for search_type in types_to_run:
        result = run_benchmark(
            args.index, 
            search_type, 
            SEARCH_TYPES[search_type]["use_vector"]
        )
        results.append(result)
    
    # Print summary if ran multiple
    if len(results) > 1:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for r in results:
            print(f"  {r['type'].upper()}: {r['recall']:.1f}% ({r['correct']}/{r['total']})")
        
        # Save results
        with open('experiments/datatalks_full_benchmark.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("\n✅ Saved to experiments/datatalks_full_benchmark.json")
