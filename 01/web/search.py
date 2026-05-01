from typing import Optional, List, Dict
from elasticsearch import Elasticsearch
from logger_config import logger, time_logger

class CourseRAGManager:
    """Manages state and connections for an Elasticsearch RAG pipeline."""
    
    def __init__(self, settings: dict):
        self.settings = settings
        self.es_client: Optional[Elasticsearch] = None
        self.index_name = self.settings.get("index_name", "course-questions")
        
    def connect_elasticsearch(self, host: str = "http://localhost:9200") -> None:
        try:
            self.es_client = Elasticsearch(host)
            if self.es_client.ping():
                logger.info("Successfully connected to Elasticsearch cluster.")
            else:
                logger.error("Elasticsearch is not responding to ping.")
                self.es_client = None
        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {e}")
            self.es_client = None

    @time_logger
    def search_faq(self, query: str) -> List[Dict]:
        if not self.es_client:
            logger.error("Search attempted without active ES connection.")
            return []
        
        # Added log to track search entry and target index
        logger.info(f"Starting ES search on index '{self.index_name}' for query: {query}")
            
        search_query = {
            "size": self.settings.get("search_size", 3),
            "query": {
                "bool": {
                    "must": {
                        "multi_match": {
                            "query": query,
                            "fields": ["question^4", "text"],
                            "type": "best_fields"
                        }
                    },
                    "filter": {
                        "term": {"course": self.settings.get("course_name")}
                    }
                }
            }
        }
        
        try:
            response = self.es_client.search(index=self.index_name, body=search_query)
            hits = response.get('hits', {}).get('hits', [])
            logger.info(f"Found {len(hits)} matching FAQ records.")
            return hits
        except Exception as e:
            logger.error(f"Elasticsearch querying failed: {e}")
            return []
