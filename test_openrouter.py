import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

print("--- DEBUG START ---")
print(f"API Key found: {api_key[:8]}...")

# Using a standard 'requests' call to see the RAW output
response = requests.post(
    url="https://openrouter.ai",
    headers={"Authorization": f"Bearer {api_key}"},
    json={
        "model": "google/gemini-2.0-flash-lite-preview-02-05:free",
        "messages": [{"role": "user", "content": "hi"}]
    }
)

print(f"Status Code: {response.status_code}")
print(f"Raw Content: {response.text}")
print("--- DEBUG END ---")
