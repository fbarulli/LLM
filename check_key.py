import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

print(f"Testing key: {api_key[:10]}...")

# 1. Include Content-Type: application/json
# 2. Add an 'HTTP-Referer' (OpenRouter likes this for ranking)
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost:3000", 
}

data = {
    "model": "google/gemini-2.0-flash-lite-preview-02-05:free",
    "messages": [{"role": "user", "content": "Say hello"}]
}

response = requests.post(
    url="https://openrouter.ai",
    headers=headers,
    json=data
)

print(f"Status Code: {response.status_code}")

try:
    json_data = response.json()
    print("✅ Success!")
    print(json_data['choices'][0]['message']['content'])
except Exception as e:
    print("❌ Still failing to get JSON.")
    print(f"First 100 chars of response: {response.text[:100]}")
