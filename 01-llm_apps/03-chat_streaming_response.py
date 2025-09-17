import os
import json
from openai import OpenAI, AzureOpenAI
from IPython.display import Markdown, display
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()
# -----------------------------
# 1. Azure OpenAI Streaming
# -----------------------------

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# Initialize conversation history
# Get user input
user_question = input("Enter your question (or 'quit' to exit): ")
if user_question.lower() == 'quit':
    print("Goodbye!")
    exit()

print(f"\nUser question: {user_question}")
print("=" * 50)

messages = [
    {
        "role": "user",
        "content": user_question
    }
]
# Azure OpenAI Streaming
print("Azure OpenAI Streaming Response:")
print("-" * 40)
stream = client.chat.completions.create(
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    messages=messages,
    stream=True  # Enable streaming
)

azure_response = ""
for chunk in stream:
    # Check if chunk has choices and if the first choice has delta content
    if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
        content = chunk.choices[0].delta.content
        print(content, end="", flush=True)
        azure_response += content

print("\n" + "=" * 50)

# -----------------------------
# 2. Gemini Streaming
# -----------------------------

# Initialize Gemini client with proper configuration
gemini_client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url=os.getenv("GEMINI_API_BASE")
)

print("Gemini 2.5 Flash Streaming Response:")
print("-" * 40)

# Create streaming request
stream = gemini_client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=messages,
    stream=True
)

gemini_response = ""
for chunk in stream:
    # Check if chunk has choices and content
    if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
        content = chunk.choices[0].delta.content
        print(content, end="", flush=True)
        gemini_response += content

print("\n" + "=" * 50)
