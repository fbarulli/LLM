import os
import sys
import logging
import json
import requests
import tiktoken
from pathlib import Path
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

# Configure logs to overwrite on every run
logging.basicConfig(
    filename='pipeline_output.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s'
)

def log_and_print(message: str, level: str):
    """
    Saves a message to the log file and prints it to the console.

    Inputs:
        message (str): The text message to record.
        level (str): The logging severity ('info' or 'error').
    
    Outputs:
        None
    """
    print(message)
    if level == "info":
        logging.info(message)
    elif level == "error":
        logging.error(message)

def load_settings(filename: str) -> dict:
    """
    Loads application parameters from an external JSON file.

    Inputs:
        filename (str): The path to the configuration file.
    
    Outputs:
        dict: A dictionary containing the system configuration.
    """
    try:
        path = Path.cwd() / filename
        if path.exists():
            with open(path, 'r') as f:
                settings = json.load(f)
                log_and_print(f"Loaded external settings from {filename}", "info")
                return settings
        else:
            raise FileNotFoundError(f"{filename} not found.")
    except Exception as e:
        log_and_print(f"Failed to parse {filename}: {e}", "error")
        sys.exit(1)

def load_secure_env() -> str:
    """
    Recursively checks folders up to 2 levels up to securely find a .env file.

    Inputs:
        None
    
    Outputs:
        str: The retrieved OpenRouter API key.
    """
    try:
        current_dir = Path.cwd()
        potential_paths = [
            current_dir / '.env',
            current_dir.parent / '.env',
            current_dir.parent.parent / '.env'
        ]
        
        env_file = next((p for p in potential_paths if p.exists()), None)
        
        if not env_file:
            raise FileNotFoundError("Could not find a .env file up to 2 directories above.")
            
        load_dotenv(dotenv_path=env_file)
        api_key = os.getenv("OPENROUTER_API_KEY")
        
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is empty or missing in the loaded .env file.")
            
        log_and_print(f"Securely loaded .env from: {env_file}", "info")
        return api_key
        
    except Exception as e:
        log_and_print(f"Security failure loading environment: {e}", "error")
        sys.exit(1)

class CourseRAGManager:
    """
    Manages state and connections for an Elasticsearch RAG pipeline.
    """
    def __init__(self, api_key: str, settings: dict):
        """
        Initializes the manager with credentials and settings dictionary.

        Inputs:
            api_key (str): OpenRouter API key.
            settings (dict): Configuration dictionary with models and tokenizers.
        """
        self.api_key = api_key
        self.settings = settings
        self.es_client = None
        self.index_name = "course-questions"
        
        try:
            self.tokenizer = tiktoken.get_encoding(self.settings.get("tokenizer_encoding"))
        except Exception as e:
            log_and_print(f"Invalid encoding in settings. Error: {e}", "error")
            sys.exit(1)
        
    def connect_elasticsearch(self, host: str):
        """
        Establishes a connection to the running Elasticsearch instance.

        Inputs:
            host (str): Network address of the database.
        
        Outputs:
            None
        """
        try:
            self.es_client = Elasticsearch(host)
            if self.es_client.ping():
                log_and_print("Successfully connected to Elasticsearch cluster.", "info")
            else:
                raise ConnectionError("Elasticsearch is not responding to ping.")
        except Exception as e:
            log_and_print(f"Failed to connect to Elasticsearch: {e}", "error")
            self.es_client = None

    def search_faq(self, query: str) -> list:
        """
        Performs a multi-match keyword search against document databases.

        Inputs:
            query (str): The search phrase typed by a user.
        
        Outputs:
            list: Raw matching hit dictionaries from Elasticsearch.
        """
        if not self.es_client:
            log_and_print("Search attempted without active ES connection.", "error")
            return []
            
        search_query = {
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
            response = self.es_client.search(index=self.index_name, body=search_query)
            hits = response.get('hits', {}).get('hits', [])
            log_and_print(f"Found {len(hits)} matching FAQ records.", "info")
            return hits
        except Exception as e:
            log_and_print(f"Elasticsearch querying failed: {e}", "error")
            return []

    def build_prompt(self, question: str, records: list) -> str:
        """
        Formats retrieved documents and queries into a full prompt string.

        Inputs:
            question (str): User's primary search question.
            records (list): Hit items returned by Elasticsearch.
        
        Outputs:
            str: Heavily structured string ready for the LLM.
        """
        context_template = "Q: {question}\nA: {text}"
        prompt_template = """
You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT:
{context}
""".strip()

        try:
            context_entries = [
                context_template.format(
                    question=hit['_source']['question'], 
                    text=hit['_source']['text']
                ) 
                for hit in records
            ]
            
            context = "\n\n".join(context_entries)
            prompt = prompt_template.format(question=question, context=context)
            
            token_count = len(self.tokenizer.encode(prompt))
            log_and_print(f"Prompt built. Length: {len(prompt)} chars | Tokens: {token_count}", "info")
            return prompt
            
        except Exception as e:
            log_and_print(f"Failed to build prompt template: {e}", "error")
            return ""

    def query_openrouter(self, prompt: str) -> str:
        """
        Sends a payload request to the OpenRouter completion endpoint.

        Inputs:
            prompt (str): Prepared payload text for the AI.
        
        Outputs:
            str: Synthesized response from the generative model.
        """
        if not prompt:
            return "Prompt was empty due to previous failures."
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referrer": "http://localhost:3000", 
        }

        data = {
            "model": self.settings.get("llm_model"),
            "messages": [{"role": "user", "content": prompt}]
        }

        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            # Check the status code first to identify 401, 404, or 500 errors
            if response.status_code != 200:
                log_and_print(f"OpenRouter API failed with code {response.status_code}. Raw response: {response.text}", "error")
                return f"Error: API returned status {response.status_code}"
            
            # Guard against empty bodies (which cause the Line 1 Column 1 error)
            if not response.text.strip():
                log_and_print("OpenRouter returned a 200 OK but the body was completely empty.", "error")
                return "Error: Server returned an empty response."

            try:
                json_data = response.json()
                answer = json_data['choices'][0]['message']['content']
                usage = json_data.get('usage', {})
                log_and_print(f"LLM Success! Total call tokens used: {usage.get('total_tokens', 'N/A')}", "info")
                return answer
            except Exception as json_err:
                log_and_print(f"Failed to parse JSON despite 200 OK. Raw body: {response.text}", "error")
                return f"Error: Invalid JSON response ({json_err})"
                
        except Exception as e:
            log_and_print(f"Fatal error during OpenRouter call: {e}", "error")
            return "Error: Could not reach completion endpoint."
