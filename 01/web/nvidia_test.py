from src.prompt_manager import query_llm
import json

with open("settings.json", "r") as f:
    settings = json.load(f)

response, provider = query_llm("Say exactly: Connection successful", settings, {"test": True})
print(f"Provider: {provider}")
print(f"Response: {response}")