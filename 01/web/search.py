import traceback
from typing import List, Dict, Any, Optional
from elasticsearch import Elasticsearch
from logger_config import logger, time_logger

class CourseRAGManager:
    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        self.es_client: Optional[Elasticsearch] = None
        self.index_name = self.settings.get("index_name", "course-questions")
        
    def connect_elasticsearch(self) -> None:
        host = self.settings.get("es_host", "http://localhost:9200")
        try:
            self.es_client = Elasticsearch(host)
            if not self.es_client.ping():
                raise ConnectionError("ES Ping failed")
        except Exception:
            logger.error(f"Connection failed: {traceback.format_exc()}")
            raise

    @time_logger
    def search_faq(self, query: str, override_size: int, course_context: Optional[str] = None) -> List[Dict]:
        """Executes search with optional course filtering."""
        search_query = {
            "size": override_size,
            "query": {
                "bool": {
                    "must": {
                        "multi_match": {
                            "query": query,
                            "fields": [
                                f"question^{self.settings.get('boost_question', 1)}",
                                f"text^{self.settings.get('boost_text', 1)}"
                            ],
                            "type": self.settings.get("search_type", "best_fields")
                        }
                    }
                }
            }
        }
        
        if course_context:
            search_query["query"]["bool"]["filter"] = {"term": {"course": course_context}}
            
        try:
            response = self.es_client.search(index=self.index_name, body=search_query)
            return response.get('hits', {}).get('hits', [])
        except Exception:
            logger.error(f"Search failed: {traceback.format_exc()}")
            return []
