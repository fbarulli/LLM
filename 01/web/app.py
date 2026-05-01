import gradio as gr
import traceback
import tiktoken
import os

from core import (
    build_prompt, 
    query_llm
)
from search import CourseRAGManager
from logger_config import logger, time_logger
# Assuming load_settings and load_secure_keys are in core.py or similar
from core import load_settings, load_secure_keys 

# 1. Configuration & Key Loading
settings: dict = load_settings(filename="settings.json")
# We now load both keys for the failover logic
nv_key, or_key = load_secure_keys()

if not nv_key and not or_key:
    logger.error("CRITICAL: No API keys found for NVIDIA or OpenRouter!")

# 2. Initialize RAG Manager (from search.py)
rag: CourseRAGManager = CourseRAGManager(settings=settings)

try:
    rag.connect_elasticsearch(host=settings.get("es_host", "http://localhost:9200"))
except Exception as e:
    logger.error(f"Failed to connect to Elasticsearch on startup: {str(e)}")

# 3. Setup Tokenizer
try:
    tokenizer_name: str = settings.get("tokenizer_encoding", "cl100k_base")
    tokenizer: tiktoken.Encoding = tiktoken.get_encoding(tokenizer_name)
except Exception:
    logger.error("Failed to load tokenizer. Defaulting to cl100k_base.")
    tokenizer = tiktoken.get_encoding("cl100k_base")

@time_logger
def glass_box_agent(user_question: str) -> tuple[str, str, str]:
    """Process question, invoke RAG, and route data with Failover LLM.
    (i) user_question / (o) raw_context_output, answer, provider_info
    """
    if not user_question.strip():
        return "Please enter a question.", "Please enter a question.", "None"

    # --- Step 1: Retrieval ---
    try:
        records: list[dict] = rag.search_faq(query=user_question)
        
        if records:
            raw_context_output: str = "\n\n".join([
                f"Document Score: {hit['_score']:.2f}\n"
                f"Q: {hit['_source']['question']}\n"
                f"A: {hit['_source']['text']}"
                for hit in records
            ])
        else:
            raw_context_output = "No matching documents found in Elasticsearch!"
            
    except Exception:
        logger.error(f"Elasticsearch retrieval error:\n{traceback.format_exc()}")
        raw_context_output = "An error occurred retrieving documents from Elasticsearch."
        records = []

    # --- Step 2: Generation (NVIDIA -> OpenRouter Failover) ---
    try:
        # Prompt building (uses tiktoken internally as per your core.py)
        prompt: str = build_prompt(
            question=user_question, 
            records=records
        )
        
        # Unified call that handles the failover logic
        answer, provider_name = query_llm(
            prompt=prompt, 
            nv_key=nv_key, 
            or_key=or_key, 
            settings=settings
        )
        
        provider_info = f"Response provided by: {provider_name}"
        return raw_context_output, answer, provider_info
        
    except Exception:
        logger.error(f"LLM Orchestration error:\n{traceback.format_exc()}")
        return raw_context_output, "An error occurred generating response.", "Error"

# 4. Gradio UI Layout
with gr.Blocks(title="Course RAG Assistant - Developer View") as demo:
    gr.Markdown("# Course FAQ Assistant")
    gr.Markdown("Search & Generation powered by NVIDIA and OpenRouter Failover.")
    
    with gr.Row():
        with gr.Column():
            user_input: gr.Textbox = gr.Textbox(
                label="Your Question", 
                placeholder="e.g., How do I execute a command in a running docker container?",
                lines=3
            )
            submit_btn: gr.Button = gr.Button("Search & Generate", variant="primary")
            provider_display = gr.Label(label="Model Provider Status")
        
    with gr.Row():
        with gr.Column():
            es_output: gr.Textbox = gr.Textbox(
                label="Raw Elasticsearch Hits (Context)", 
                lines=12,
                interactive=False
            )
        
        with gr.Column():
            llm_output: gr.Textbox = gr.Textbox(
                label="Final Answer", 
                lines=12,
                interactive=False
            )
    
    # Setup actions
    input_list = [user_input]
    output_list = [es_output, llm_output, provider_display]

    submit_btn.click(fn=glass_box_agent, inputs=input_list, outputs=output_list)
    user_input.submit(fn=glass_box_agent, inputs=input_list, outputs=output_list)
    
    gr.Markdown("---")
    gr.Markdown("Check `pipeline_output.log` and `metrics.log` for execution durations and token usage.")

if __name__ == "__main__":
    logger.info("Starting local Gradio web server with failover logic...")
    demo.launch(server_name="127.0.0.1", server_port=7870)
