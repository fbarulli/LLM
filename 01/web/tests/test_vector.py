import sys
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')

from src.search import CourseRAGManager
from src.config_manager import load_config

print('=' * 60)
print('TESTING VECTOR SEARCH')
print('=' * 60)

try:
    settings = load_config('../settings.json')
    manager = CourseRAGManager(settings)
    manager.connect_elasticsearch()

    test_queries = [
        'When does the course start?',
        'What are the prerequisites?',
        'How do I join the course?',
    ]

    print('\n📝 RESULTS:')
    for query in test_queries:
        results = manager.search_faq(query, 3, None)
        if results:
            top = results[0]['_source']
            print(f'\n   Query: "{query}"')
            print(f'   → {top["course"]}: {top["question"][:50]}...')
        else:
            print(f'\n   Query: "{query}" → No results')
except Exception as e:
    print(f'Error: {e}')
