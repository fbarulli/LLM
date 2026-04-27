import sys
import json
import requests
from elasticsearch import Elasticsearch
from core import load_settings, log_and_print

def fetch_raw_data(data_url: str) -> list:
    """
    Downloads the raw JSON file from a specific URL.

    Inputs:
        data_url (str): The internet address of the raw FAQ JSON repository.
    
    Outputs:
        list: The parsed raw data array.
    """
    try:
        response = requests.get(url=data_url, timeout=30)
        if response.status_code == 200:
            log_and_print(f"Successfully downloaded data from {data_url}", "info")
            return response.json()
        else:
            raise requests.ConnectionError(f"HTTP code {response.status_code}")
    except Exception as e:
        log_and_print(f"Failed to fetch raw data: {e}", "error")
        sys.exit(1)

def transform_documents(raw_data: list) -> list:
    """
    Restructures the multi-tiered course list into single-document dictionaries 
    containing flat course labels using clean list comprehensions.

    Inputs:
        raw_data (list): The original multi-course structure fetched from the web.
    
    Outputs:
        list: A flattened array of individual question-answer dictionaries.
    """
    try:
        # Outer comprehension iterates over courses, inner comprehension iterates over its documents
        flattened_records = [
            {
                "text": doc.get("text"),
                "section": doc.get("section"),
                "question": doc.get("question"),
                "course": course.get("course")
            }
            for course in raw_data
            for doc in course.get("documents", [])
        ]
        
        log_and_print(f"Transformed data into {len(flattened_records)} individual flat documents.", "info")
        return flattened_records
    except Exception as e:
        log_and_print(f"Failed to transform documents: {e}", "error")
        sys.exit(1)

def setup_index_and_ingest(es_client: Elasticsearch, index_name: str, records: list):
    """
    Creates an optimized search index mapping and executes data ingestion.

    Inputs:
        es_client (Elasticsearch): Active, connected Python elasticsearch instance.
        index_name (str): Desired string label for the search database.
        records (list): The completed flat document array to push.
    
    Outputs:
        None
    """
    try:
        # Check if the index already exists; delete if true to fulfill clean slate runs
        if es_client.indices.exists(index=index_name):
            es_client.indices.delete(index=index_name)
            log_and_print(f"Existing index '{index_name}' dropped for a fresh run.", "info")

        index_settings = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0
            },
            "mappings": {
                "properties": {
                    "text": {"type": "text"},
                    "section": {"type": "keyword"},
                    "question": {"type": "text"},
                    "course": {"type": "keyword"}
                }
            }
        }

        es_client.indices.create(index=index_name, body=index_settings)
        log_and_print(f"Elasticsearch index '{index_name}' successfully built with active mappings.", "info")

        # Push data sequentially into the new index
        indexed_count = 0
        for doc in records:
            es_client.index(index=index_name, document=doc)
            indexed_count += 1

        log_and_print(f"Ingestion process finished. Pushed {indexed_count} documents into '{index_name}'.", "info")
        
    except Exception as e:
        log_and_print(f"Failed during Elasticsearch index setup or ingestion: {e}", "error")
        sys.exit(1)

if __name__ == "__main__":
    log_and_print("Starting automated data ingestion pipeline...", "info")
    
    # Extract truth solely from your JSON settings
    settings = load_settings(filename="settings.json")
    
    # Connect directly to your local instance
    es_client = Elasticsearch("http://localhost:9200")
    if not es_client.ping():
        log_and_print("Elasticsearch is not responding at http://localhost:9200. Please start your container.", "error")
        sys.exit(1)

    # Official DataTalksClub raw FAQ repository link
    data_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main/01-intro/documents.json?raw=1'

    
    # Run the pipeline
    raw_data = fetch_raw_data(data_url=data_url)
    flattened_docs = transform_documents(raw_data=raw_data)
    
    # CourseRAGManager in core assumes the index name is "course-questions"
    setup_index_and_ingest(
        es_client=es_client, 
        index_name="course-questions", 
        records=flattened_docs
    )
    
    log_and_print("Ingestion script completed without failures.", "info")
