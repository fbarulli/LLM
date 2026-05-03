import json
from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")
index = "course-questions"

# 1. Pick a query that we KNOW exists in multiple courses (like 'Slack')
test_query = "Slack"

# 2. Run a PURE GLOBAL search (No filters at all)
query = {
    "size": 10,
    "query": {
        "multi_match": {
            "query": test_query,
            "fields": ["question", "text"]
        }
    }
}

res = es.search(index=index, body=query)
courses_found = [hit['_source']['course'] for hit in res['hits']['hits']]

print(f"--- 🛰️ Global Search Test for '{test_query}' ---")
print(f"Courses found in Top 10: {set(courses_found)}")
print(f"Raw results: {courses_found}")
