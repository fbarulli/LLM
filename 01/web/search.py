import traceback
import json
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
        """Diagnostic search function."""
        if not self.es_client:
            return []

        mm_query = {
            "multi_match": {
                "query": query,
                "fields": [
                    f"question^{self.settings.get('boost_question', 1)}",
                    f"text^{self.settings.get('boost_text', 1)}"
                ],
                "type": self.settings.get("search_type", "best_fields")
            }
        }

        # --- THE TRUTH PRINTS ---
        print(f"\n🔍 [DEBUG] Query: '{query[:30]}...' | Context: {course_context}")

        if course_context:
            final_query = {"bool": {"must": mm_query, "filter": {"term": {"course": course_context}}}}
        else:
            final_query = mm_query

        try:
            response = self.es_client.search(index=self.index_name, query=final_query, size=override_size)
            hits = response.get('hits', {}).get('hits', [])
            
            if not course_context and hits:
                courses = [h['_source']['course'] for h in hits]
                print(f"📊 [DEBUG] Diversity: {set(courses)}")
                
            return hits
        except Exception:
            logger.error(f"Search failed: {traceback.format_exc()}")
            return []
