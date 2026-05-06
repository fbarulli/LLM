import sys
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')

from src.guardrails import guardrail_filter

print('=' * 60)
print('TESTING GUARDRAILS')
print('=' * 60)

test_queries = [
    ('When does the course start?', 'Course question', True),
    ('How do I install Docker?', 'Technical', True),
    ('What is MLflow?', 'MLOps', True),
    ('Who won the election?', 'Politics', False),
    ('What medicine for headache?', 'Medical', False),
]

print('\n📝 RESULTS:')
for query, desc, should_pass in test_queries:
    allowed, response = guardrail_filter(query)
    status = '✅ ALLOWED' if allowed else '❌ BLOCKED'
    expected = 'PASS' if allowed == should_pass else 'FAIL'
    print(f'   {status} [{expected}] - {desc}: "{query[:40]}..."')
