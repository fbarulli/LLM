import sys
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')
import json
import re

# Load hard eval results
with open('../experiments/results/hybrid_bm25_hard_full_with_quality.json', 'r') as f:
    data = json.load(f)

# Technical keywords
tech_keywords = [
    'docker', 'kubernetes', 'container', 'pip', 'install', 'import',
    'error', 'bug', 'fail', 'debug', 'code', 'script', 'function',
    'api', 'endpoint', 'command', 'terminal', 'bash', 'config',
    'python', 'jupyter', 'notebook', 'git', 'github', 'mlflow',
    'spark', 'airflow', 'terraform', 'aws', 'gcp', 'azure'
]

technical_queries = []
other_queries = []

for r in data['results']:
    if r.get('k') == 5:
        query = r['query'].lower()
        is_tech = any(kw in query for kw in tech_keywords)
        if is_tech:
            technical_queries.append(r)
        else:
            other_queries.append(r)

print(f"Technical queries: {len(technical_queries)}")
print(f"Other queries: {len(other_queries)}")
print(f"Total: {len(technical_queries) + len(other_queries)}")

print("\n" + "=" * 60)
print("TECHNICAL QUERIES IN HARD EVAL SET")
print("=" * 60)

for r in technical_queries[:10]:
    status = "✅" if r.get('relevant') else "❌"
    print(f"{status} {r['query']}")
