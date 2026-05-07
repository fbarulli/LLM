#!/usr/bin/env python
"""Run evaluation pipeline on all ES variations."""

import sys
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')

import json
from elasticsearch import Elasticsearch
from src.search import CourseRAGManager
from src.core import generate_document_id
from tqdm import tqdm
import os

def get_eval_set():
    """Get all documents from Elasticsearch as eval set."""
    es = Elasticsearch("http://localhost:9200")
    response = es.search(index="faqs_complete", size=1000, query={"match_all": {}})
    
    eval_set = []
    for hit in response['hits']['hits']:
        source = hit['_source']
        eval_set.append({
            "query": source['question'],
            "course": source['course'],
            "expected_id": hit['_id']
        })
    return eval_set

def run_config(config_path, eval_set):
    """Run a single config and return recall."""
    with open(config_path, 'r') as f:
        settings = json.load(f)
    
    settings["es_host"] = "http://localhost:9200"
    
    manager = CourseRAGManager(settings)
    manager.connect_elasticsearch()
    
    correct = 0
    for item in tqdm(eval_set, desc=settings['name'], leave=False):
        results = manager.search_faq(item['query'], 5, item['course'])
        if item['expected_id'] in [hit['_id'] for hit in results]:
            correct += 1
    
    recall = correct / len(eval_set) * 100
    return settings['name'], recall, correct

def main():
    print("=" * 60)
    print("EVALUATION PIPELINE - ALL ES VARIATIONS")
    print("=" * 60)
    
    # Get eval set
    eval_set = get_eval_set()
    print(f"\n📊 Eval set: {len(eval_set)} queries")
    
    # Config files to run
    config_dir = "experiments/configs"
    configs = [
        "bm25_default.json",
        "bm25_high_question.json", 
        "bm25_high_text.json",
        "bm25_balanced.json",
        "vector_default.json",
        "hybrid_default.json",
        "hybrid_balanced.json"
    ]
    
    results = []
    for config_file in configs:
        config_path = os.path.join(config_dir, config_file)
        if os.path.exists(config_path):
            name, recall, correct = run_config(config_path, eval_set)
            results.append((name, recall, correct))
            print(f"✅ {name}: {correct}/{len(eval_set)} = {recall:.1f}%")
    
    # Summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    results.sort(key=lambda x: x[1], reverse=True)
    for name, recall, correct in results:
        print(f"  {name:25}: {recall:.1f}%")
    
    # Save results
    with open('experiments/results/all_variations.json', 'w') as f:
        json.dump([{"name": n, "recall": r, "correct": c, "total": len(eval_set)} for n, r, c in results], f, indent=2)
    print("\n✅ Saved to experiments/results/all_variations.json")

if __name__ == "__main__":
    main()
