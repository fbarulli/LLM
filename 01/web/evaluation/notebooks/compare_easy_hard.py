# /home/admin/LLM/LLM/01/web/notebooks/compare_easy_hard.py

import sys
import json
import hashlib
import logging
import traceback
import os
import pandas as pd
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, '/home/admin/LLM/LLM/01/web')
from src.visualizer import RAGVisualizer

def generate_doc_id(doc: Dict[str, Any]) -> str:
    """Exact hashing logic used in the Zoomcamp ground truth generation."""
    # Note: If your ground truth was made with 'llm-zoomcamp', but the JSON says 'data-engineering',
    # the hashes will never match.
    course = doc.get('course', 'data-engineering-zoomcamp')
    question = doc.get('question', '')
    text = doc.get('text', '')
    combined = f"{course}-{question}-{text}"
    return hashlib.md5(combined.encode('utf-8')).hexdigest()

def get_doc_text(doc_id: str, flattened_docs: Dict[str, Dict[str, Any]]) -> str:
    """Lookup that checks for exact hash match."""
    doc = flattened_docs.get(doc_id)
    if doc:
        q = doc.get('question', 'N/A')[:60]
        a = doc.get('text', 'N/A')[:60]
        return f"Q: {q}...\n  A: {a}..."
    return f"ID {doc_id} not found in current memory map."

# --- DATA LOADING ---
# Try to load both files and merge them
doc_paths: List[str] = ['../documents-llm.json', 'documents.json']
flattened_docs: Dict[str, Dict[str, Any]] = {}

for path in doc_paths:
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            # Standard Zoomcamp JSON structure: list of {course: ..., documents: [...]}
            for course_entry in data:
                course_name = course_entry.get('course', '')
                for doc in course_entry.get('documents', []):
                    doc['course'] = course_name
                    # RE-HASH every document to ensure we have a fresh mapping
                    did = generate_doc_id(doc)
                    flattened_docs[did] = doc
            
            logger.info(f"Loaded/Re-hashed docs from {path}. Total in map: {len(flattened_docs)}")
        except Exception as e:
            logger.error(f"Error reading {path}: {e}")

# DEBUG PRINT: Show one generated hash and its source to compare manually
if flattened_docs:
    first_id = list(flattened_docs.keys())[0]
    sample = flattened_docs[first_id]
    logger.info(f"DEBUG HASH: {first_id} (Source: {sample['course']} | {sample['question'][:30]}...)")

viz = RAGVisualizer()
registry = viz.get_experiment_registry()
filenames = registry['filename'].tolist()
df = viz.load_selected_experiments(filenames)

# --- RECALL COMPARISON ---
easy_runs = [f for f in filenames if 'hard' not in f]
hard_runs = [f for f in filenames if 'hard' in f]

print("\n" + "="*60)
print("📊 RECALL@5 COMPARISON")
print("="*60)

for f_name in easy_runs + hard_runs:
    label = f_name.replace('.json', '')
    m = (df['run_label'] == label) & (df['k'] == 5)
    recall = df.loc[m, 'success'].mean() if not df.loc[m].empty else 0.0
    print(f"  {f_name:30} : {recall:.2%}")

# --- FAILURE ANALYSIS ---
target = "baseline_bm25_hard.json"
try:
    with open(f"experiments/results/{target}", 'r') as f:
        res = json.load(f)
    
    res_df = pd.DataFrame(res['results'])
    fails = res_df.loc[(res_df['k'] == 5) & (res_df['success'] == False)].head(3)

    print("\n" + "="*60)
    print(f"🔍 FAILURE ANALYSIS: {target}")
    print("="*60)

    for _, row in fails.iterrows():
        print(f"\nQUERY: \"{row['query']}\"")
        print(f"EXPECTED TEXT ({row['expected_id']}):\n  {get_doc_text(row['expected_id'], flattened_docs)}")
        print(f"\nFOUND TEXT    ({row['found_id']}):\n  {get_doc_text(row['found_id'], flattened_docs)}")
        print("-" * 40)
except Exception as e:
    logger.error(f"Deep dive failed: {e}")
