#!/usr/bin/env python
"""Add faithfulness and relevancy metrics to ALL result files."""

import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import litellm
import re
from tqdm import tqdm
from elasticsearch import Elasticsearch

def get_document_text(es_client, index_name, doc_id):
    """Fetch document text from Elasticsearch by ID."""
    try:
        response = es_client.get(index=index_name, id=doc_id)
        return response['_source'].get('text', '')[:300]
    except:
        return ''

def batch_evaluate_quality(queries_data, batch_size=10):
    """Evaluate answer quality in batches."""
    results = []
    
    for i in range(0, len(queries_data), batch_size):
        batch = queries_data[i:i + batch_size]
        
        prompt = """Rate each query-response pair. Return JSON array with 'relevant' (bool) and 'faithful' (bool).

"""
        for j, q in enumerate(batch):
            prompt += f"""{j+1}. Q:{q['query'][:80]} | A:{q['response'][:150]}\n"""
        
        prompt += "\nReturn: [{\"relevant\": true/false, \"faithful\": true/false}, ...]"
        
        try:
            response = litellm.completion(
                model="nvidia_nim/nvidia/nemotron-mini-4b-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=500
            )
            result_text = response.choices[0].message.content
            json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
            if json_match:
                batch_results = json.loads(json_match.group())
                results.extend(batch_results)
            else:
                results.extend([{'faithful': False, 'relevant': False} for _ in batch])
        except Exception as e:
            print(f"Batch error: {e}")
            results.extend([{'faithful': False, 'relevant': False} for _ in batch])
    
    return results

def add_quality_to_file(result_path, es_client, index_name):
    """Add quality metrics to a single result file."""
    
    with open(result_path, 'r') as f:
        data = json.load(f)
    
    # Get results for K=5 with found_id
    results = [r for r in data['results'] 
               if r.get('k') == 5 and r.get('found_id') and r.get('found_id') != 'NONE']
    
    if not results:
        print(f"   Skipping {os.path.basename(result_path)} - no valid results")
        return None
    
    # Fetch response text for each result
    queries_data = []
    for r in results:
        response_text = get_document_text(es_client, index_name, r['found_id'])
        if response_text:
            queries_data.append({
                'query': r['query'],
                'response': response_text
            })
        else:
            queries_data.append({
                'query': r['query'],
                'response': ''
            })
    
    # Evaluate
    evaluations = batch_evaluate_quality(queries_data)
    
    # Add metrics to results
    for r, eval_result in zip(results, evaluations):
        r['faithful'] = eval_result.get('faithful', False)
        r['relevant'] = eval_result.get('relevant', False)
    
    # Save
    output_path = result_path.replace('.json', '_with_quality.json')
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    return output_path

def main():
    print("=" * 60)
    print("ADDING QUALITY METRICS TO ALL RESULT FILES")
    print("=" * 60)
    
    # Connect to Elasticsearch
    es_client = Elasticsearch("http://localhost:9200")
    index_name = "course-questions"
    
    results_dir = "experiments/results"
    result_files = [f for f in os.listdir(results_dir) 
                    if f.endswith('.json') and 'with_quality' not in f]
    
    print(f"\n📊 Found {len(result_files)} result files")
    
    for fname in tqdm(result_files, desc="Processing"):
        filepath = os.path.join(results_dir, fname)
        output = add_quality_to_file(filepath, es_client, index_name)
        if output:
            print(f"   ✅ Created: {os.path.basename(output)}")
    
    print("\n" + "=" * 60)
    print("✅ Quality metrics added to all files")
    print("=" * 60)

if __name__ == "__main__":
    main()
