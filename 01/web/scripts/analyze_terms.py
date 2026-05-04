# /home/admin/LLM/LLM/01/web/analyze_terms.py

import json
from collections import Counter
from nltk.corpus import stopwords
import nltk

# Download stopwords if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

with open("documents.json", "r") as f:
    data = json.load(f)

for course in data:
    print(f"\n=== {course['course']} ===")
    
    all_words = []
    for doc in course['documents']:
        # Clean the question
        question = doc['question']
        if " - " in question:
            question = question.split(" - ", 1)[1].strip()
        
        # Split and filter
        words = question.lower().split()
        words = [w for w in words if w not in stop_words and len(w) > 2]
        all_words.extend(words)
        
        # Also analyze text field (first 200 chars)
        text_words = doc['text'].lower().split()[:50]
        text_words = [w for w in text_words if w not in stop_words and len(w) > 2]
        all_words.extend(text_words)
    
    top_terms = Counter(all_words).most_common(20)
    print(f"Top meaningful terms:")
    for term, count in top_terms:
        print(f"  {term}: {count}")