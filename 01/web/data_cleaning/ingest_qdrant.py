"""
data_cleaning/ingest_qdrant.py
===============================
Ingests clean documents into Qdrant vector database.

Creates a collection with question vectors for semantic search.
Uses the same embedding model as the ES vector search.

Input:  data_cleaning/data/processed/clean.jsonl
Output: Qdrant collection 'faqs'

Run:    uv run python data_cleaning/ingest_qdrant.py
"""
import os
import json
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

INPUT = 'data_cleaning/data/processed/clean.jsonl'
COLLECTION_NAME = 'faqs'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
QDRANT_HOST = 'localhost'
QDRANT_PORT = 6333
VECTOR_SIZE = 384  # all-MiniLM-L6-v2 dimension


def main():
    # Load embedding model
    print(f'Loading embedding model: {EMBEDDING_MODEL}')
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Connect to Qdrant
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    print(f'Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}')

    # Delete existing collection if present
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
        print(f'Deleted existing collection: {COLLECTION_NAME}')

    # Create collection
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f'Created collection: {COLLECTION_NAME}')

    # Load documents and create embeddings
    points = []
    total = 0
    
    with open(INPUT) as f:
        for line in f:
            doc = json.loads(line)
            
            # Create embedding from question text
            embedding = model.encode(doc['question']).tolist()
            
            points.append(PointStruct(
                id=total,
                vector=embedding,
                payload={
                    'es_id': doc['id'],
                    'question': doc['question'],
                    'answer': doc['answer'],
                    'course': doc['course'],
                    'section': doc['section'],
                }
            ))
            total += 1
            
            if total % 100 == 0:
                print(f'  Encoded {total} documents...')

    # Batch upload
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
    
    print(f'Ingested {total} documents into Qdrant')


if __name__ == '__main__':
    main()
