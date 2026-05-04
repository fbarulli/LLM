# /home/admin/LLM/LLM/01/web/src/prompt_manager.py

import os
import json
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from litellm import completion
from src.logger_config import logger, time_logger

def load_secure_keys():
    """Load API keys from .env file"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(current_dir, "..", ".env")
    
    if os.path.exists(env_path):
        load_dotenv(dotenv_path=env_path)
    
    nvidia_key = os.getenv("NVIDIA_NIM_API_KEY", "")
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    
    langfuse_public = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    langfuse_secret = os.getenv("LANGFUSE_SECRET_KEY", "")
    langfuse_host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    
    # Set environment variables for litellm
    if nvidia_key:
        os.environ["NVIDIA_NIM_API_KEY"] = nvidia_key
    if openrouter_key:
        os.environ["OPENROUTER_API_KEY"] = openrouter_key
    os.environ["OR_SITE_URL"] = "http://localhost:3000"
    
    if langfuse_public and langfuse_secret:
        os.environ["LANGFUSE_PUBLIC_KEY"] = langfuse_public
        os.environ["LANGFUSE_SECRET_KEY"] = langfuse_secret
        os.environ["LANGFUSE_HOST"] = langfuse_host
    
    return nvidia_key, openrouter_key

def build_prompt(question: str, records: List[Dict]) -> str:
    """Builds a prompt from retrieved documents for LLM."""
    context_template = "Q: {question}\nA: {text}"
    prompt_template = """You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT:
{context}"""
    
    context_entries = []
    for hit in records:
        context_entries.append(
            context_template.format(
                question=hit['_source']['question'],
                text=hit['_source']['text']
            )
        )
    
    context = "\n\n".join(context_entries)
    prompt = prompt_template.format(question=question, context=context)
    return prompt

@time_logger
def query_llm_provider(prompt: str, model_name: str, provider_prefix: str, metadata: dict = None) -> Optional[str]:
    """Generic LLM provider call using litellm."""
    if metadata is None:
        metadata = {}
    
    try:
        response = completion(
            model=f"{provider_prefix}/{model_name}",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024,
            timeout=60,
            metadata=metadata
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Provider {provider_prefix}/{model_name} failed: {e}")
        return None

def query_llm(prompt: str, settings: dict, metadata: dict = None) -> Tuple[str, str]:
    """Orchestrator for LLM queries based on settings.json."""
    if metadata is None:
        metadata = {}
    
    nv_key, or_key = load_secure_keys()
    active_providers = []
    
    # Build provider list based on settings toggles
    if settings.get("use_nvidia"):
        active_providers.append({
            "model": settings.get("nvidia_model"),
            "prefix": "nvidia_nim",
            "label": "NVIDIA_LLAMA",
            "cost": settings.get("nvidia_cost_1k", 0)
        })
    
    if settings.get("use_openrouter"):
        active_providers.append({
            "model": settings.get("openrouter_model"),
            "prefix": "openrouter",
            "label": "OPENROUTER",
            "cost": settings.get("openrouter_cost_1k", 0)
        })
    
    if not active_providers:
        logger.error("No LLM providers enabled in settings")
        return "Error: No LLM providers enabled.", "NONE"
    
    for provider in active_providers:
        logger.info(f"Requesting {provider['label']} ({provider['model']})...")
        
        meta = {**metadata, "cost_per_1k_tokens": provider["cost"], "model": provider["model"]}
        
        answer = query_llm_provider(
            prompt, 
            provider["model"], 
            provider["prefix"], 
            meta
        )
        
        if answer:
            logger.info(f"Success from {provider['label']}")
            return answer, provider["label"]
        
        logger.warning(f"Provider {provider['label']} failed, trying next...")
    
    return "Error: All enabled LLM providers failed.", "NONE"