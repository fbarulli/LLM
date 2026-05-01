import json
import os
import litellm
from typing import Optional, Tuple
from litellm import completion
from dotenv import load_dotenv
from logger_config import logger, time_logger
from prompt_manager import build_prompt

def track_usage(kwargs, response_obj, start_time, end_time):
    usage = getattr(response_obj, 'usage', {})
    model = kwargs["model"]
    logger.info(f"LLM Success | Model: {model} | Tokens: {usage} | Latency: {end_time - start_time:.2f}s")

litellm.success_callback = [track_usage]

def load_settings(filename="settings.json"):
    try:
        with open(filename, "r") as f:
            settings = json.load(f)
            return settings
    except Exception as e:
        logger.error(f"Failed to load settings: {e}")
        raise

def load_secure_keys():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(current_dir, "..", "..", ".env")
    
    if os.path.exists(env_path):
        load_dotenv(dotenv_path=env_path)
    
    os.environ["NVIDIA_NIM_API_KEY"] = os.getenv("NVIDIA_API_KEY", "")
    os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY", "")
    os.environ["OR_SITE_URL"] = "http://localhost:3000"
    
    os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY", "")
    os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_HOST", "https://langfuse.com")
        
    return os.environ["NVIDIA_NIM_API_KEY"], os.environ["OPENROUTER_API_KEY"]

@time_logger
def query_llm_provider(prompt: str, model_name: str, provider_prefix: str, metadata: dict) -> Optional[str]:
    try:
        response = completion(
            model=f"{provider_prefix}/{model_name}",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024,
            timeout=60,
            metadata=metadata
        )
        # Fix: Access the first choice in the list
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Provider {provider_prefix} failed for model {model_name}: {e}")
        return None


@time_logger
def query_llm(prompt: str, nv_key: str, or_key: str, settings: dict, metadata: dict) -> Tuple[str, str]:
    active_providers = []
    
    if settings["use_nvidia"]:
        meta = {**metadata, "cost_1k": settings["nvidia_cost_1k"]}
        active_providers.append((settings["nvidia_model"], "nvidia_nim", "NVIDIA_LLAMA", meta))
    
    if settings["use_mistral_nvidia"]:
        meta = {**metadata, "cost_1k": settings["mistral_cost_1k"]}
        active_providers.append((settings["mistral_model"], "nvidia_nim", "NVIDIA_MISTRAL", meta))

    if settings["use_nemotron_nvidia"]:
        meta = {**metadata, "cost_1k": settings["nemotron_cost_1k"]}
        active_providers.append((settings["nemotron_model"], "nvidia_nim", "NVIDIA_NEMOTRON", meta))

    if settings["use_openrouter"]:
        meta = {**metadata, "cost_1k": settings["openrouter_cost_1k"]}
        active_providers.append((settings["openrouter_model"], "openrouter", "OPENROUTER", meta))

    for model, prefix, label, meta_with_cost in active_providers:
        logger.info(f"Orchestrator: Requesting {label}...")
        answer = query_llm_provider(prompt, model, prefix, meta_with_cost)
        if answer: 
            return answer, label
        logger.warning(f"Orchestrator: {label} failed.")

    logger.error("Orchestrator: All enabled providers failed.")
    return "Error: All enabled LLM providers failed.", "NONE"
