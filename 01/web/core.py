import requests
from typing import Optional, Tuple
from logger_config import logger, time_logger

def build_prompt(question: str, records: list) -> str:
    """
    Formats retrieved documents and queries into a full prompt string.
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
        return prompt
        
    except Exception as e:
        logger.error(f"Failed to build prompt template: {e}")
        raise

@time_logger
def query_nvidia(prompt: str, api_key: str, settings: dict) -> Optional[str]:
    """Uses the exact integrate.api.nvidia.com URL from settings."""
    url = settings["nvidia_url"]
    model_name = settings["nvidia_model"]
    
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.70,
        "max_tokens": 1024,
        "stream": False
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        
        logger.warning(f"NVIDIA Error {response.status_code}: {response.text[:100]}")
        return None
    except Exception as e:
        logger.error(f"NVIDIA call failed: {e}")
        return None

@time_logger
def query_openrouter(prompt: str, api_key: str, settings: dict) -> Optional[str]:
    """Uses the exact openrouter.ai/api/v1 URL from settings."""
    url = settings["openrouter_url"]
    model_name = settings["openrouter_model"]
    
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:3000",
        "Accept": "application/json"
    }

    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        
        logger.error(f"OpenRouter Error {response.status_code}")
        return None
    except Exception as e:
        logger.error(f"OpenRouter call failed: {e}")
        return None

def query_llm(prompt: str, nv_key: str, or_key: str, settings: dict) -> Tuple[str, str]:
    """Orchestrator strictly following JSON toggle switches."""
    if settings.get("use_nvidia"):
        answer = query_nvidia(prompt, nv_key, settings)
        if answer: return answer, "NVIDIA"

    if settings.get("use_openrouter"):
        answer = query_openrouter(prompt, or_key, settings)
        if answer: return answer, "OpenRouter"

    return "Error: All enabled LLM providers failed.", "NONE"
