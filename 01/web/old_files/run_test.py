import os
import time
from datetime import datetime
from dotenv import load_dotenv
import litellm

def run_test():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.abspath(os.path.join(current_dir, "..", "..", ".env"))
    load_dotenv(dotenv_path=env_path)

    model_id = "nvidia_nim/nvidia/nemotron-mini-4b-instruct"
    litellm.model_cost[model_id] = {
        "max_tokens": 4096, 
        "input_cost_per_token": 0.0, 
        "output_cost_per_token": 0.0
    }

    litellm.success_callback = ["langfuse"]
    os.environ["LANGFUSE_PUBLIC_KEY"] = os.environ.get("LANGFUSE_PUBLIC_KEY")
    os.environ["LANGFUSE_SECRET_KEY"] = os.environ.get("LANGFUSE_SECRET_KEY")
    os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com"
    
    # Create a clean timestamp string
    my_timestamp = datetime.now().strftime("%H:%M:%S")
    trace_name = f"test_{my_timestamp}"
    
    print(f"Creating trace: {trace_name}")
    
    response = litellm.completion(
        model=model_id,
        messages=[{"role": "user", "content": f"Test at {my_timestamp}"}],
        mock_response=f"Response at {my_timestamp}",
        metadata={
            "trace_name": trace_name,
            "tags": [f"time-{my_timestamp}"],
            "trace_metadata": {
                "ping_time": my_timestamp,
                "unix_time": int(time.time())
            }
        }
    )
    
    print(f"Done! Trace '{trace_name}' created at {my_timestamp}")
    print("Check dashboard in 5-10 min: https://cloud.langfuse.com")

if __name__ == "__main__":
    run_test()