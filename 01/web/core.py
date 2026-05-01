import time
from typing import Optional, Tuple
from litellm import completion
from logger_config import logger, time_logger
from langfuse.decorators import observe, langfuse_context

@observe()
@time_logger
def query_llm_provider(prompt: str, model_name: str, provider_prefix: str, metadata: dict, tags: list) -> Optional[str]:
    short_name = model_name.split('/')[-1][:20]
    
    langfuse_context.update_current_trace(
        name=f"LLM_{provider_prefix}_{short_name}",
        metadata={**metadata, "model": model_name, "provider": provider_prefix, "timestamp": time.time()},
        tags=tags
    )

    try:
        response = completion(
            model=f"{provider_prefix}/{model_name}",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024,
            timeout=60
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Provider {provider_prefix}/{model_name} failed: {e}")
        return None

@observe()
@time_logger
def query_llm(prompt: str, settings: dict, metadata: dict, tags: list) -> Tuple[str, str]:
    from langfuse_config import get_providers
    
    providers = get_providers(settings)
    
    if not providers:
        return "Error: No providers configured.", "NONE"
    
    for model, prefix, label, cost in providers:
        meta = {**metadata, "cost_per_1k_tokens": cost}
        logger.info(f"Requesting {label} ({model})...")
        answer = query_llm_provider(prompt, model, prefix, meta, tags)
        
        if answer:
            langfuse_context.update_current_trace(
                metadata={"successful_provider": label}
            )
            langfuse_context.flush()
            return answer, label
    
    return "Error: All enabled providers failed.", "NONE"