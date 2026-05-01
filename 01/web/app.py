import gradio as gr
import traceback
import tiktoken
import os

from core import (
    build_prompt, 
    query_llm,
    load_settings, 
    load_secure_keys 
)
from search import CourseRAGManager
from logger_config import logger, time_logger

settings: dict = load_settings(filename="settings.json")

nv_key, or_key = load_secure_keys()

if not nv_key and not or_key:
    logger.error("CRITICAL: No API keys found for NVIDIA or OpenRouter!")

rag: CourseRAGManager = CourseRAGManager(settings=settings)

try:
    rag.connect_elasticsearch(host=settings["es_host"])
except Exception as e:
    logger.error(f"Failed to connect to Elasticsearch on startup: {str(e)}")

try:
    tokenizer_name: str = settings["tokenizer_encoding"]
    tokenizer: tiktoken.Encoding = tiktoken.get_encoding(tokenizer_name)
except Exception:
    logger.error("Failed to load tokenizer from settings. Defaulting to cl100k_base.")
    tokenizer = tiktoken.get_encoding("cl100k_base")

@time_logger
def glass_box_agent(user_question: str) -> tuple[str, str, str]:
    if not user_question.strip():
        return "Please enter a question.", "Please enter a question.", "None"

    try:
        records: list[dict] = rag.search_faq(query=user_question)
        doc_ids = [hit.get("_id", "unknown") for hit in records]
        
        if records:
            raw_context_output: str = "\n\n".join([
                f"Document Score: {hit['_score']:.2f}\n"
                f"Q: {hit['_source']['question']}\n"
                f"A: {hit['_source']['text']}"
                for hit in records
            ])
        else:
            raw_context_output = "No matching documents found in Elasticsearch!"
            doc_ids = []
            
    except Exception:
        logger.error(f"Elasticsearch retrieval error:\n{traceback.format_exc()}")
        raw_context_output = "An error occurred retrieving documents from Elasticsearch."
        records = []
        doc_ids = []

    try:
        prompt: str = build_prompt(
            question=user_question, 
            records=records
        )
        
        metadata = {
            "document_ids": doc_ids,
            "course": settings["course_name"]
        }

        answer, provider_name = query_llm(
            prompt=prompt, 
            nv_key=nv_key, 
            or_key=or_key, 
            settings=settings,
            metadata=metadata
        )
        
        provider_info = f"Response provided by: {provider_name}"
        return raw_context_output, answer, provider_info
        
    except Exception:
        logger.error(f"LLM Orchestration error:\n{traceback.format_exc()}")
        return raw_context_output, "An error occurred generating response.", "Error"

with gr.Blocks(title="Course RAG Assistant - Developer View") as demo:
    gr.Markdown("# Course FAQ Assistant")
    gr.Markdown("Search & Generation powered by NVIDIA and OpenRouter.")
    
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
    
    input_list = [user_input]
    output_list = [es_output, llm_output, provider_display]

    submit_btn.click(fn=glass_box_agent, inputs=input_list, outputs=output_list)
    user_input.submit(fn=glass_box_agent, inputs=input_list, outputs=output_list)
    
    gr.Markdown("---")
    gr.Markdown(f"Course Context: {settings['course_name']}")

if __name__ == "__main__":
    logger.info("Starting local Gradio web server...")
    demo.launch(server_name="127.0.0.1", server_port=7870)
