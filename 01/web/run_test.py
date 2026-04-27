from core import CourseRAGManager, load_settings, load_secure_env

settings = load_settings()
api_key = load_secure_env()

rag = CourseRAGManager(api_key=api_key, settings=settings)
rag.connect_elasticsearch()

# Quick console test
records = rag.search_faq("How do I run docker?")
prompt = rag.build_prompt("How do I run docker?", records)
answer = rag.query_openrouter(prompt)

print(f"Test Answer: {answer}")
