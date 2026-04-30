import os
import requests
import tiktoken
import json
from pathlib import Path
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from typing import Optional

# Functional Import: Offloading logging and timing collision handlers
from logger_config import logger, time_logger

def load_settings(filename: str) -> dict:
    """Loads application parameters from an external JSON file.
    (i) filename / (o) settings
    """
    path: Path = Path(__file__).resolve().parent / filename
    
    try:
        with open(path, 'r') as f:
            settings: dict = json.load(f)
            logger.info(f"Loaded external settings from {filename}")
            return settings
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {filename}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON syntax in file: {filename}")
        raise

def load_secure_env() -> str:
    """Recursively checks parent directories to find a .env file.
    (i) None / (o) api_key
    """
    current: Path = Path(__file__).resolve().parent
    
    for _ in range(3):
        env_path: Path = current / '.env'
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            api_key: Optional[str] = os.getenv("OPENROUTER_API_KEY")
            
            if not api_key:
                logger.error("OPENROUTER_API_KEY is empty in the loaded .env file.")
                raise ValueError("API key missing in environment file.")
                
            logger.info("Securely loaded environment variables.")
            return api_key
            
        current = current.parent
        
    logger.error("Could not find a .env file up to 2 directories above.")
    raise FileNotFoundError("Environment file not found.")

@time_logger
def build_prompt(question: str, records: list, tokenizer: tiktoken.Encoding) -> str:
    """Formats retrieved documents and queries into a full prompt string.
    (i) question, records, tokenizer / (o) prompt
    """
    context_template: str = "Q: {question}\nA: {text}"
    prompt_template: str = """
You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT:
{context}
""".strip()

    try:
        context_entries: list[str] = [
            context_template.format(
                question=hit['_source']['question'], 
                text=hit['_source']['text']
            ) 
            for hit in records
        ]
        
        context: str = "\n\n".join(context_entries)
        prompt: str = prompt_template.format(question=question, context=context)
        
        token_count: int = len(tokenizer.encode(prompt))
        logger.info(f"Prompt built. Length: {len(prompt)} chars | Tokens: {token_count}")
        return prompt
        
    except Exception:
        logger.error("Failed to build prompt template.")
        raise


@time_logger
def query_openrouter(prompt: str, api_key: str, model_name: str) -> str:
    """Sends a payload request to the OpenRouter completion endpoint.
    (i) prompt, api_key, model_name / (o) response_str
    """
    if not prompt:
        return "Prompt was empty due to previous failures."
        
    headers: dict[str, str] = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referrer": "http://localhost:3000", 
    }

    data: dict = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response: requests.Response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code != 200:
            logger.error(f"OpenRouter API failed with code {response.status_code}. Raw response: {response.text}")
            return f"Error: API returned status {response.status_code}"
        
        if not response.text.strip():
            logger.error("OpenRouter returned a 200 OK but the body was completely empty.")
            return "Error: Server returned an empty response."

        try:
            json_data: dict = response.json()
            answer: str = json_data['choices'][0]['message']['content']
            usage: dict = json_data.get('usage', {})
            logger.info(f"LLM Success! Total call tokens used: {usage.get('total_tokens', 'N/A')}")
            return answer
        except Exception as json_err:
            logger.error(f"Failed to parse JSON despite 200 OK. Raw response content: {response.text}")
            return f"Error: Invalid JSON response ({json_err})"
            
    except Exception as e:
        logger.error(f"Fatal error during OpenRouter call: {str(e)}")
        return "Error: Could not reach completion endpoint."


class CourseRAGManager:
    """Manages state and connections for an Elasticsearch RAG pipeline."""
    
    def __init__(self, settings: dict):
        """Initializes the manager with credentials and settings dictionary.
        (i) settings / (o) None
        """
        self.settings: dict = settings
        self.es_client: Optional[Elasticsearch] = None
        self.index_name: str = self.settings.get("index_name", "course-questions")
        
    def connect_elasticsearch(self, host: str) -> None:
        """Establishes a connection to the running Elasticsearch instance.
        (i) host / (o) None
        """
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
    def search_faq(self, query: str) -> list[dict]:
        """Performs a multi-match keyword search against document databases.
        (i) query / (o) records
        """
        if not self.es_client:
            logger.error("Search attempted without active ES connection.")
            return []
            
        search_query: dict = {
            "size": self.settings.get("search_size"),
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
            response: dict = self.es_client.search(index=self.index_name, body=search_query)
            hits: list[dict] = response.get('hits', {}).get('hits', [])
            logger.info(f"Found {len(hits)} matching FAQ records.")
            return hits
        except Exception as e:
            logger.error(f"Elasticsearch querying failed: {e}")
            return []
