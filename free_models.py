import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

response = requests.get(
    url="https://openrouter.ai/api/v1/models",
    headers={"Authorization": f"Bearer {api_key}"}
)

models = response.json()["data"]
free_models = [m["id"] for m in models if ":free" in m["id"]]

for m in free_models:
    print(m)