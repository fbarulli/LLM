import gradio as gr
from core import (
    CourseRAGManager, 
    load_settings, 
    load_secure_env, 
    build_prompt, 
    query_openrouter
)
from logger_config import logger, time_logger
import traceback
import tiktoken

settings: dict = load_settings(filename="settings.json")
api_key: str | None = load_secure_env()

if not api_key:
    logger.error("CRITICAL: Secure environment API key not found!")

rag: CourseRAGManager = CourseRAGManager(settings=settings)

try:
    rag.connect_elasticsearch(host="http://localhost:9200")
except Exception as e:
    logger.error(f"Failed to connect to Elasticsearch on startup: {str(e)}")

try:
    tokenizer_name: str = settings.get("tokenizer_encoding", "cl100k_base")
    tokenizer: tiktoken.Encoding = tiktoken.get_encoding(tokenizer_name)
except Exception:
    logger.error("Failed to load tokenizer. Defaulting to cl100k_base.")
    tokenizer = tiktoken.get_encoding("cl100k_base")

@time_logger
def glass_box_agent(user_question: str) -> tuple[str, str]:
    """Process question, invoke RAG, and route data.
    (i) user_question / (o)raw_context_output, answer
    """
    if not user_question.strip():
        return "Please enter a question.", "Please enter a question."

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

    try:
        if not api_key:
            return raw_context_output, "LLM cannot be reached. API key is missing."
            
        prompt: str = build_prompt(
            question=user_question, 
            records=records, 
            tokenizer=tokenizer
        )
        
        answer: str = query_openrouter(
            prompt=prompt, 
            api_key=api_key, 
            model_name=settings.get("llm_model", "openai/gpt-4o-mini")
        )
        
        return raw_context_output, answer
        
    except Exception:
        logger.error(f"OpenRouter LLM error:\n{traceback.format_exc()}")
        return raw_context_output, "An error occurred generating response from the LLM."

with gr.Blocks(title="Course RAG Assistant - Developer View") as demo:
    gr.Markdown("# Course FAQ Assistant")
    gr.Markdown("Type a question below to see what Elasticsearch finds and how the LLM responds.")
    
    with gr.Row():
        with gr.Column():
            user_input: gr.Textbox = gr.Textbox(
                label="Your Question", 
                placeholder="e.g., How do I execute a command in a running docker container?",
                lines=3
            )
            submit_btn: gr.Button = gr.Button("Search & Generate", variant="primary")
        
    with gr.Row():
        with gr.Column():
            es_output: gr.Textbox = gr.Textbox(
                label="Raw Elasticsearch Hits (Context Provided to LLM)", 
                lines=12,
                interactive=False
            )
        
        with gr.Column():
            llm_output: gr.Textbox = gr.Textbox(
                label="Final LLM Answer", 
                lines=12,
                interactive=False
            )
    
    submit_btn.click(
        fn=glass_box_agent,
        inputs=[user_input],
        outputs=[es_output, llm_output]
    )
    
    user_input.submit(
        fn=glass_box_agent,
        inputs=[user_input],
        outputs=[es_output, llm_output]
    )
    
    gr.Markdown("---")
    gr.Markdown("Check `pipeline_output.log` to audit behind-the-scenes metrics and token overhead.")

if __name__ == "__main__":
    logger.info("Starting local Gradio web server...")
    demo.launch(server_name="127.0.0.1")
