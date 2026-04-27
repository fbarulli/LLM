import gradio as gr
from core import CourseRAGManager, load_settings, load_secure_env, logging, log_and_print
import traceback

# Initialize system operations from core with explicit arguments
settings = load_settings(filename="settings.json")
api_key = load_secure_env()
rag = CourseRAGManager(api_key=api_key, settings=settings)

# Point to your local elasticsearch instance explicitly
rag.connect_elasticsearch(host="http://localhost:9200")

def glass_box_agent(user_question: str):
    """
    Receives user UI input, invokes the RAG pipeline, and routes data to multiple display fields.

    Inputs:
        user_question (str): The raw string typed by a user in Gradio.
    
    Outputs:
        tuple: A string of formatted raw context, followed by the actual LLM string.
    """
    try:
        # Step 1: Search Elasticsearch
        records = rag.search_faq(query=user_question)
        
        if records:
            
            # Changed hit['_score']['question'] to hit['_source']['question'] 
            # to prevent potential KeyError.
            raw_context_output = "\n\n".join([
                f"Document Score: {hit['_score']:.2f}\n"
                f"Q: {hit['_source']['question']}\n"
                f"A: {hit['_source']['text']}"
                for hit in records
            ])
        else:
            raw_context_output = "No matching documents found in Elasticsearch!"
        
        # Step 2: Build Prompt & Query LLM
        prompt = rag.build_prompt(question=user_question, records=records)
        answer = rag.query_openrouter(prompt=prompt)
        
        return raw_context_output, answer
        
    except Exception as e:
        
        error_trace = traceback.format_exc()
        logging.error(f"UI Error processing request:\n{error_trace}")
        
        # Friendly UI return
        return "An error occurred retrieving docs. Check pipeline_output.log for details.", "An error occurred generating response."

# Drawing the visual interface
with gr.Blocks(title="Course RAG Assistant - Developer View") as demo:
    gr.Markdown("# Course FAQ Assistant")
    gr.Markdown("Type a question below to see what Elasticsearch finds and how the LLM responds.")
    
    with gr.Row():
        with gr.Column():
            user_input = gr.Textbox(
                label="Your Question", 
                placeholder="e.g., How do I execute a command in a running docker container?",
                lines=3
            )
            submit_btn = gr.Button("Search & Generate", variant="primary")
        
    with gr.Row():
        with gr.Column():
            es_output = gr.Textbox(
                label="Raw Elasticsearch Hits (Context Provided to LLM)", 
                lines=12,
                interactive=False
            )
        
        with gr.Column():
            llm_output = gr.Textbox(
                label="Final LLM Answer", 
                lines=12,
                interactive=False
            )
    
    # Map visual triggers
    submit_btn.click(
        fn=glass_box_agent,
        inputs=[user_input],
        outputs=[es_output, llm_output]
    )
    
    gr.Markdown("---")
    gr.Markdown("Check `pipeline_output.log` to audit behind-the-scenes metrics and token overhead.")

if __name__ == "__main__":
    log_and_print("Starting local Gradio web server...", "info")
    demo.launch(server_name="127.0.0.1", server_port=7860)
