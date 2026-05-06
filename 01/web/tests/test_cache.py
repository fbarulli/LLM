import sys
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')
import time

from src.search import CourseRAGManager
from src.config_manager import load_config

print('=' * 60)
print('TESTING CACHE')
print('=' * 60)

try:
    settings = load_config('../experiments/configs/vector_bm25_cached.json')
    manager = CourseRAGManager(settings)
    manager.connect_elasticsearch()

    query = 'When does the course start?'

    print('\n📝 FIRST CALL (Cache Miss):')
    start = time.time()
    results = manager.search_faq(query, 3, None)
    first_time = (time.time() - start) * 1000
    print(f'   Time: {first_time:.2f}ms')
    print(f'   Response: {results[0]["_source"]["text"][:80]}...')

    print('\n📝 SECOND CALL (Cache Hit):')
    start = time.time()
    results = manager.search_faq(query, 3, None)
    second_time = (time.time() - start) * 1000
    print(f'   Time: {second_time:.2f}ms')
    print(f'   Response: {results[0]["_source"]["text"][:80]}...')

    print(f'\n📊 Speedup: {first_time/second_time:.2f}x')
except Exception as e:
    print(f'Error: {e}')
    print('Make sure Redis is running and config file exists')
