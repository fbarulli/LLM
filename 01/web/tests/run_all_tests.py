import subprocess
import sys
import os

os.chdir('/home/admin/LLM/LLM/01/web/tests')

tests = [
    ('test_guardrails.py', 'Guardrails'),
    ('test_vector.py', 'Vector Search'),
    ('test_comparison.py', 'BM25 vs Vector'),
    ('test_cache.py', 'Cache'),
]

print('\n' + '=' * 60)
print('RUNNING ALL TESTS')
print('=' * 60)

for test_file, name in tests:
    if not os.path.exists(test_file):
        print(f'\n⚠️ Skipping {name}: {test_file} not found')
        continue
    
    print(f'\n🔍 Testing {name}...')
    result = subprocess.run([sys.executable, test_file], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr and 'Error' in result.stderr:
        print(f'Errors: {result.stderr[:200]}')

print('\n✅ All tests completed')
