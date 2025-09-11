import os
from openai import OpenAI, AzureOpenAI
from IPython.display import Markdown, display
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# -----------------------------
# 1. Azure OpenAI Test
# -----------------------------

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

question = "Please ask a question to assess someone's IQ. Respond only with the question."
messages = [{"role": "user", "content": question}]
# ask it using Azure OpenAI
response = client.chat.completions.create(
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    messages=messages
)

question = response.choices[0].message.content
print(question)
# form a new messages list
messages = [{"role": "user", "content": question}]

response = client.chat.completions.create(
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    messages=messages
)
azureOpenAIResponse = response.choices[0].message.content
print("Azure OpenAI 4o-mini: "+azureOpenAIResponse)

# -----------------------------
# 2. Gemini Test
# -----------------------------

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url=os.getenv("GEMINI_API_BASE")
)

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[{"role": "user", "content": question}]
)
geminiResponse= response.choices[0].message.content
print("Gemini 2.5 Flash: "+geminiResponse)


