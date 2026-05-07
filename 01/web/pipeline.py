#!/usr/bin/env python
"""Main pipeline: test or benchmark DataTalks dataset."""

import sys
import json
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')

from src.search import CourseRAGManager
from src.core import generate_document_id
from tqdm import tqdm

SEARCH_TYPES = {
    "bm25": {"use_vector": False, "boost_question": 20.0, "boost_text": 1.0},
    "vector": {"use_vector": True, "boost_question": 1.0, "boost_text": 1.0},
    "hybrid": {"use_vector": True, "boost_question": 20.0, "boost_text": 1.0}
}

def load_documents():
    with open('documents_datatalks.json', 'r') as f:
        data = json.load(f)
    return data[0]['documents']

def load_eval_set(docs):
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

def run_benchmark(search_type, index_name, eval_set):
    print(f"\n{'='*50}")
    print(f"Benchmark: {search_type.upper()}")
    print('='*50)
    
    config = SEARCH_TYPES[search_type]
    settings = {
        "es_host": "http://localhost:9200",
        "index_name": index_name,
        "search_type": search_type,
        "use_vector": config["use_vector"],
        "embedding_model": "all-MiniLM-L6-v2",
        "boost_question": config["boost_question"],
        "boost_text": config["boost_text"]
    }
    
    manager = CourseRAGManager(settings)
    manager.connect_elasticsearch()
    
    correct = 0
    for item in tqdm(eval_set, desc="Processing"):
        results = manager.search_faq(item['query'], 5, item['course'])
        if item['expected_id'] in [hit['_id'] for hit in results]:
            correct += 1
    
    recall = correct / len(eval_set) * 100
    print(f"Recall@5: {correct}/{len(eval_set)} = {recall:.1f}%")
    return recall, correct

def run_quick_test(docs, index_name):
    print("=" * 60)
    print("QUICK TEST: BM25 vs VECTOR vs HYBRID")
    print("=" * 60)
    
    test_doc = docs[0]
    query = test_doc['question']
    expected_id = generate_document_id({
        "text": test_doc['text'],
        "question": query,
        "course": "datatalks-zoomcamp"
    })
    
    print(f"\nTest query: {query[:60]}...")
    
    for search_type in SEARCH_TYPES:
        config = SEARCH_TYPES[search_type]
        settings = {
            "es_host": "http://localhost:9200",
            "index_name": index_name,
            "search_type": search_type,
            "use_vector": config["use_vector"],
            "embedding_model": "all-MiniLM-L6-v2"
        }
        
        manager = CourseRAGManager(settings)
        manager.connect_elasticsearch()
        
        results = manager.search_faq(query, 5, "datatalks-zoomcamp")
        success = expected_id in [hit['_id'] for hit in results]
        print(f"  {search_type.upper()}: {'✅' if success else '❌'}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--index", default="datatalks-faqs")
    parser.add_argument("--type", choices=list(SEARCH_TYPES.keys()))
    
    args = parser.parse_args()
    
    docs = load_documents()
    print(f"📄 Documents: {len(docs)}")
    
    if args.mode == "quick":
        run_quick_test(docs, args.index)
    else:
        eval_set = load_eval_set(docs)
        print(f"📊 Eval set: {len(eval_set)} queries")
        
        if args.type:
            recall, correct = run_benchmark(args.type, args.index, eval_set)
        else:
            results = {}
            for search_type in SEARCH_TYPES:
                recall, correct = run_benchmark(search_type, args.index, eval_set)
                results[search_type] = recall
            
            print("\n" + "=" * 60)
            print("FINAL SUMMARY")
            print("=" * 60)
            for st, recall in results.items():
                print(f"  {st.upper()}: {recall:.1f}%")
            
            with open('experiments/datatalks_full_benchmark.json', 'w') as f:
                json.dump(results, f, indent=2)
            print("\n✅ Saved to experiments/datatalks_full_benchmark.json")
