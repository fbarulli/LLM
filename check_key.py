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
    "HTTP-Referrer": "http://localhost:3000", 
}

data = {
"model": "google/gemma-3n-e2b-it:free",

"messages": [{"role": "user", "content": "Say hello"}]
}

response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers=headers,
    json=data
)


print(f"Status Code: {response.status_code}")

if response.status_code == 200:
    try:
        json_data = response.json()
        print("✅ Success!")
        print(json_data['choices'][0]['message']['content'])
    except Exception as e:
        print(f"❌ Failed to parse JSON: {e}")
else:
    print(f"❌ Request failed with status {response.status_code}")
    print(response.text)
