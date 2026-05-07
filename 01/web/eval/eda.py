# /home/admin/LLM/LLM/01/web/eval/eda.py

import sys
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from eval.eval_set import get_eval_set_from_es
import re

def remove_stopwords(text):
    stopwords = {'the', 'to', 'i', 'in', 'how', 'a', 'is', 'for', 'and', 'not', 
                 'of', 'on', 'with', 'it', 'you', 'that', 'this', 'are', 'be', 
                 'at', 'by', 'from', 'or', 'as', 'what', 'why', 'when', 'where',
                 'which', 'who', 'whom', 'do', 'does', 'did', 'have', 'has', 'had',
                 'can', 'could', 'will', 'would', 'should', 'may', 'might', 'my', 'your'}
    words = text.lower().split()
    return [w for w in words if w not in stopwords and len(w) > 2]

def analyze_dataset():
    eval_set = get_eval_set_from_es()
    
    queries = []
    courses = []
    question_lengths = []
    word_counts = []
    unique_words_per_query = []
    technical_terms = []
    
    technical_patterns = {
        'Kafka': r'\bkafka\b',
        'Docker': r'\bdocker\b',
        'Terraform': r'\bterraform\b',
        'AWS': r'\baws\b',
        'GCP': r'\bgcp\b|google cloud',
        'BigQuery': r'\bbigquery\b',
        'PySpark': r'\bpyspark\b',
        'MLflow': r'\bmlflow\b',
        'WandB': r'\bwandb\b',
        'dbt': r'\bdbt\b',
        'PostgreSQL': r'\bpostgres\b',
        'Redis': r'\bredis\b',
        'Kubernetes': r'\bkubernetes\b|k8s',
        'Git': r'\bgit\b',
        'Python': r'\bpython\b',
        'pandas': r'\bpandas\b',
        'numpy': r'\bnumpy\b'
    }
    
    for item in eval_set:
        doc = item['original_doc']
        question = doc.get('question', '')
        course = doc.get('course', '')
        
        queries.append(question)
        courses.append(course)
        question_lengths.append(len(question))
        
        words = question.split()
        word_counts.append(len(words))
        unique_words_per_query.append(len(set(words)))
        
        term_count = 0
        for tech, pattern in technical_patterns.items():
            if re.search(pattern, question, re.IGNORECASE):
                term_count += 1
                technical_terms.append(tech)
    
    df = pd.DataFrame({
        'query': queries,
        'course': courses,
        'length': question_lengths,
        'word_count': word_counts,
        'unique_words': unique_words_per_query
    })
    
    print("=== DATASET OVERVIEW ===")
    print(f"Total queries: {len(df)}")
    print(f"Unique courses: {df['course'].nunique()}")
    print(f"\nCourse distribution (top 10):")
    course_counts = df['course'].value_counts()
    for course, count in course_counts.head(10).items():
        print(f"  {course}: {count}")
    
    print("\n=== TECHNICAL TERM DETECTION ===")
    tech_counter = Counter(technical_terms)
    print(f"Queries with technical terms: {len(technical_terms)}/{len(df)} = {len(technical_terms)/len(df)*100:.1f}%")
    print("Top technical terms:")
    for term, count in tech_counter.most_common(10):
        print(f"  {term}: {count}")
    
    print("\n=== CONTENT RICHNESS ANALYSIS ===")
    content_words = []
    for query in queries:
        content_words.extend(remove_stopwords(query))
    
    content_word_freq = Counter(content_words)
    print(f"Total unique content words (after stopword removal): {len(content_word_freq)}")
    print("Top content words:")
    for word, count in content_word_freq.most_common(20):
        print(f"  '{word}': {count}")
    
    print("\n=== QUERY COMPLEXITY METRICS ===")
    content_word_ratio = df['unique_words'].values / df['word_count'].values
    print(f"Avg unique word ratio: {content_word_ratio.mean():.2f}")
    print(f"Queries with high uniqueness (>0.8): {sum(content_word_ratio > 0.8)}/{len(df)} = {sum(content_word_ratio > 0.8)/len(df)*100:.1f}%")
    
    return df

def analyze_course_alignment():
    results_dir = '/home/admin/LLM/LLM/01/web/experiments/results'
    
    with open(f'{results_dir}/bm25_default.json', 'r') as f:
        data = json.load(f)
    
    results = [r for r in data['results'] if r['k'] == 5]
    
    same_course = 0
    different_course = 0
    missing_course = 0
    
    for r in results:
        expected_course = r.get('found_course', '')
        if expected_course == 'NONE':
            missing_course += 1
        elif r['success']:
            same_course += 1
        else:
            different_course += 1
    
    print("\n=== COURSE ALIGNMENT ANALYSIS (BM25 Default) ===")
    print(f"Correct document from same course: {same_course}/{len(results)} = {same_course/len(results)*100:.1f}%")
    print(f"Success but different course: {different_course}/{len(results)} = {different_course/len(results)*100:.1f}%")
    print(f"Missing course info: {missing_course}/{len(results)}")
    
    return same_course, different_course

def analyze_error_cases():
    results_dir = '/home/admin/LLM/LLM/01/web/experiments/results'
    
    with open(f'{results_dir}/vector_default.json', 'r') as f:
        data = json.load(f)
    
    failures = [r for r in data['results'] if r['k'] == 5 and not r['success']]
    
    print(f"\n=== VECTOR FAILURE ANALYSIS ===")
    print(f"Total failures at K=5: {len(failures)}/{len([r for r in data['results'] if r['k'] == 5])}")
    
    if failures:
        print("\nSample failures:")
        for i, failure in enumerate(failures[:5]):
            query = failure['query'][:80]
            print(f"{i+1}. {query}...")
            print(f"   Expected ID: {failure['expected_id'][:20]}...")
            print(f"   Found ID: {failure['found_id'][:20]}...")
    
    return failures

if __name__ == "__main__":
    df = analyze_dataset()
    same_course, different_course = analyze_course_alignment()
    failures = analyze_error_cases()
