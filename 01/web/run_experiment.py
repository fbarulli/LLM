#!/usr/bin/env python
"""Run experiment with config overrides via command line."""

import sys
import json
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')

from src.search import CourseRAGManager
from src.config_manager import load_config

def run(index_name, search_type, query="How to compute quantile?", size=5):
    settings = {
        "es_host": "http://localhost:9200",
        "index_name": index_name,
        "search_type": search_type,
        "use_vector": search_type in ["vector", "hybrid"],
        "embedding_model": "all-MiniLM-L6-v2",
        "boost_question": 20.0,
        "boost_text": 1.0
    }
    
    print(f"Index: {index_name}")
    print(f"Search type: {search_type}")
    print(f"Query: {query}")
    
    manager = CourseRAGManager(settings)
    manager.connect_elasticsearch()
    
    results = manager.search_faq(query, size, None)
    
    print(f"\nTop {len(results)} results:")
    for i, hit in enumerate(results):
        print(f"{i+1}. {hit['_source']['course']}: {hit['_source']['question'][:60]}...")
    
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="datatalks-faqs")
    parser.add_argument("--type", choices=["bm25", "vector", "hybrid"], default="vector")
    parser.add_argument("--query", default="How to compute quantile?")
    parser.add_argument("--size", type=int, default=5)
    
    args = parser.parse_args()
    run(args.index, args.type, args.query, args.size)
