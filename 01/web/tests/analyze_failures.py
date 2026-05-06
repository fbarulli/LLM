import sys
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')
import json
import pandas as pd

# Load hard eval results with quality
with open('../experiments/results/hybrid_bm25_hard_full_with_quality.json', 'r') as f:
    data = json.load(f)

# Find failures (relevant = false)
failures = []
for r in data['results']:
    if r.get('k') == 5 and r.get('relevant') == False:
        failures.append({
            'query': r['query'],
            'expected_course': r['expected_course'],
            'found_course': r['found_course'],
            'found_question': r.get('found_question', 'N/A'),
            'score': r.get('score', 0)
        })

print(f"Total failures: {len(failures)}")
print("\n" + "=" * 60)
print("FAILURE PATTERNS")
print("=" * 60)

# Group by type of question
question_types = {}
for f in failures:
    # Extract question type from query
    q_lower = f['query'].lower()
    if 'start' in q_lower or 'begin' in q_lower:
        q_type = 'start_date'
    elif 'prerequisite' in q_lower or 'need to know' in q_lower:
        q_type = 'prerequisites'
    elif 'submit' in q_lower or 'homework' in q_lower:
        q_type = 'homework'
    elif 'certificate' in q_lower or 'graduate' in q_lower:
        q_type = 'certificate'
    elif 'join' in q_lower or 'sign up' in q_lower:
        q_type = 'registration'
    else:
        q_type = 'other'
    
    question_types.setdefault(q_type, []).append(f['query'])

for q_type, queries in question_types.items():
    print(f"\n{q_type.upper()}: {len(queries)} failures")
    for q in queries[:3]:
        print(f"  - {q[:70]}...")

# Show specific examples
print("\n" + "=" * 60)
print("SAMPLE FAILURES (first 10)")
print("=" * 60)
for f in failures[:10]:
    print(f"\nQuery: {f['query']}")
    print(f"  Expected course: {f['expected_course']}")
    print(f"  Found course: {f['found_course']}")
