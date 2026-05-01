import gradio as gr
import traceback
import tiktoken
import os
from langfuse_config import load_settings, load_api_keys
from core import query_llm
from prompt_manager import build_prompt
from search import CourseRAGManager
from logger_config import logger, time_logger
from langfuse.decorators import observe

settings = load_settings(filename="settings.json")
nv_key, or_key = load_api_keys()

rag = CourseRAGManager(settings=settings)
try:
    rag.connect_elasticsearch(host=settings["es_host"])
except Exception as e:
    logger.error(f"Failed to connect to Elasticsearch: {str(e)}")

try:
    tokenizer = tiktoken.get_encoding(settings["tokenizer_encoding"])
except Exception:
    tokenizer = tiktoken.get_encoding("cl100k_base")

@observe()
@time_logger
def glass_box_agent(user_question: str) -> tuple[str, str, str]:
    if not user_question.strip():
        return "Please enter a question.", "Please enter a question.", "None"

    # --- Step 1: Retrieval ---
    try:
        records = rag.search_faq(query=user_question)
        doc_ids = [hit.get("_id", "unknown") for hit in records]
        
        if records:
            raw_context_output = "\n\n".join([
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
        return "An error occurred during retrieval.", "Search Error", "Error"

    # --- Step 2: Generation ---
    try:
        prompt = build_prompt(question=user_question, records=records)
        
        metadata = {
            "document_ids": doc_ids,
            "course": settings["course_name"]
        }
        
        tags = [settings["course_name"], "dev_test"]

        answer, provider_name = query_llm(
            prompt=prompt,
            settings=settings,
            metadata=metadata,
            tags=tags
        )
        
        return raw_context_output, answer, f"Response provided by: {provider_name}"
    except Exception:
        logger.error(f"LLM Orchestration error:\n{traceback.format_exc()}")
        return raw_context_output, "An error occurred during generation.", "Error"

with gr.Blocks(title="Course RAG Assistant") as demo:
    gr.Markdown(f"# {settings['course_name']} FAQ Assistant")
    
    with gr.Row():
        with gr.Column():
            user_input = gr.Textbox(label="Your Question", placeholder="Ask anything...", lines=3)
            submit_btn = gr.Button("Search & Generate", variant="primary")
            provider_display = gr.Label(label="Model Provider Status")
        
    with gr.Row():
        es_output = gr.Textbox(label="Raw Elasticsearch Hits (Context)", lines=12, interactive=False)
        llm_output = gr.Textbox(label="Final Answer", lines=12, interactive=False)
    
    submit_btn.click(fn=glass_box_agent, inputs=[user_input], outputs=[es_output, llm_output, provider_display])
    user_input.submit(fn=glass_box_agent, inputs=[user_input], outputs=[es_output, llm_output, provider_display])
    
    gr.Markdown("---")
    gr.Markdown(f"System Context: {settings['course_name']} | Powered by LiteLLM & Langfuse")

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7870)