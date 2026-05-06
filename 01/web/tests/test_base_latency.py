import litellm
import time
import os

print("Testing NVIDIA API base latency...")

# Simple completion test
messages = [{"role": "user", "content": "Say 'OK'"}]

for i in range(3):
    start = time.time()
    try:
        response = litellm.completion(
            model="nvidia_nim/meta/llama-3.1-8b-instruct",
            messages=messages,
            max_tokens=10,
            temperature=0
        )
        elapsed = time.time() - start
        print(f"Call {i+1}: {elapsed:.2f}s - Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"Call {i+1}: ERROR - {e}")

print("\nTesting with longer prompt (similar to evaluation)...")
long_prompt = """Evaluate if the RESPONSE answers the QUESTION and if it's faithful to the CONTEXT.

QUESTION: When does the course start?
CONTEXT: The course starts on March 15th, 2024. You can register at any time.
RESPONSE: The course starts on March 15th.

Answer with JSON only: {"relevant": true, "faithful": true}"""

start = time.time()
response = litellm.completion(
    model="nvidia_nim/meta/llama-3.1-8b-instruct",
    messages=[{"role": "user", "content": long_prompt}],
    max_tokens=100,
    temperature=0
)
elapsed = time.time() - start
print(f"Long prompt: {elapsed:.2f}s")
print(f"Response: {response.choices[0].message.content[:100]}")
