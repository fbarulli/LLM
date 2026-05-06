import sys
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')
import json
from src.core import generate_document_id

print("Fixing expanded eval set...")

# Load expanded queries
with open('../experiments/expanded_eval_set.json', 'r') as f:
    expanded = json.load(f)

# Load original documents to find correct IDs
with open('../documents.json', 'r') as f:
    data = json.load(f)

# Build a mapping from question to document ID
question_to_id = {}
for course in data:
    course_name = course['course']
    for doc in course['documents']:
        question = doc['question']
        if " - " in question:
            question = question.split(" - ", 1)[1].strip()
        
        clean_doc = {
            "text": doc['text'],
            "question": question,
            "course": course_name
        }
        doc_id = generate_document_id(clean_doc)
        question_to_id[question] = doc_id

# Add expected_id to each expanded query
fixed = []
for item in expanded:
    original_q = item['original_question']
    expected_id = question_to_id.get(original_q)
    if expected_id:
        fixed.append({
            'query': item['query'],
            'expected_id': expected_id,
            'expected_course': item['expected_course']
        })
    else:
        print(f"Warning: No ID found for: {original_q[:50]}...")

print(f"Fixed {len(fixed)} queries out of {len(expanded)}")
print(f"Missing: {len(expanded) - len(fixed)}")

# Save fixed version
with open('../experiments/expanded_fixed.json', 'w') as f:
    json.dump(fixed, f, indent=2)

# Merge with hard eval set
with open('../experiments/hard_eval_set.json', 'r') as f:
    hard_eval = json.load(f)

hard_queries = []
for item in hard_eval:
    for variant in item['paraphrased_queries']:
        hard_queries.append({
            'query': variant,
            'expected_id': item['expected_id'],
            'expected_course': item['expected_course']
        })

merged = hard_queries + fixed
print(f"\nTotal merged queries: {len(merged)} (hard: {len(hard_queries)}, expanded: {len(fixed)})")

# Save merged
with open('../experiments/merged_fixed_190.json', 'w') as f:
    json.dump(merged, f, indent=2)

print("✅ Saved to experiments/merged_fixed_190.json")
