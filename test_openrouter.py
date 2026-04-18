from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv
import os

load_dotenv()

# If you haven't set the env var yet, you can also pass it directly for a quick test:
# llm = ChatOpenRouter(openrouter_api_key="your_key", model="google/gemini-2.0-flash-001")

llm = ChatOpenRouter(model="google/gemma-4-26b-a4b-it:free")

try:
    res = llm.invoke("Say 'LangChain is ready!'")
    print(res.content)
except Exception as e:
    print(f"Error: {e}")
