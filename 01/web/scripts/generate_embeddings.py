# /home/admin/LLM/LLM/01/web/scripts/generate_embeddings.py

import sys
import os
import json
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from tqdm import tqdm

sys.path.insert(0, '/home/admin/LLM/LLM/01/web')
from src.config_manager import load_config

def main():
    print("=" * 60)
    print("GENERATING EMBEDDINGS FOR VECTOR SEARCH")
    print("=" * 60)
    
    # Load model
    model_name = "all-MiniLM-L6-v2"
    print(f"\n🤖 Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"   Model dimension: {model.get_sentence_embedding_dimension()}")
    
    # Connect to Elasticsearch
    settings = load_config("experiments/configs/baseline_bm25.json")
    es_client = Elasticsearch(settings.get("es_host", "http://localhost:9200"))
    
    # Create new index with vector field
    vector_index = "course-questions-vector"
    
    # Delete if exists
    if es_client.indices.exists(index=vector_index):
        es_client.indices.delete(index=vector_index)
        print(f"🗑️ Deleted existing index: {vector_index}")
    
    # Create index with dense vector mapping
    mapping = {
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "text": {"type": "text"},
                "section": {"type": "keyword"},
                "question": {"type": "text"},
                "course": {"type": "keyword"},
                "question_vector": {
                    "type": "dense_vector",
                    "dims": 384,
                    "index": True,
                    "similarity": "cosine"
                }
            }
        },
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        }
    }
    
    es_client.indices.create(index=vector_index, body=mapping)
    print(f"✅ Created index: {vector_index} with vector field")
    
    # Load documents
    with open("documents.json", "r") as f:
        data = json.load(f)
    
    # Flatten documents
    documents = []
    for course in data:
        course_name = course["course"]
        for doc in course["documents"]:
            # Clean question
            question = doc["question"]
            if " - " in question:
                question = question.split(" - ", 1)[1].strip()
            
            documents.append({
                "text": doc["text"],
                "section": doc.get("section", "General"),
                "question": question,
                "course": course_name
            })
    
    print(f"\n📄 Loaded {len(documents)} documents")
    
    # Generate embeddings and index
    print("\n🔢 Generating embeddings and indexing...")
    
    for i, doc in enumerate(tqdm(documents)):
        # Generate embedding from question + text
        content = f"{doc['question']} {doc['text'][:500]}"
        embedding = model.encode(content).tolist()
        
        # Create document ID
        from src.core import generate_document_id
        doc_id = generate_document_id(doc)
        
        # Index with vector
        es_client.index(
            index=vector_index,
            id=doc_id,
            document={
                "id": doc_id,
                "text": doc["text"],
                "section": doc["section"],
                "question": doc["question"],
                "course": doc["course"],
                "question_vector": embedding
            }
        )
    
    print(f"\n✅ Indexed {len(documents)} documents with embeddings")
    print(f"   Index name: {vector_index}")
    print(f"   Model: {model_name}")

if __name__ == "__main__":
    main()