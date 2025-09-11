import os
import json
from openai import OpenAI, AzureOpenAI
from IPython.display import Markdown, display
from dotenv import load_dotenv


competitors = []
answers = []



# Load environment variables from .env file
load_dotenv()
# -----------------------------
# 1. Azure OpenAI Question Generation
# -----------------------------

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)
# Inference intensive so will take time
# question = "Provide a challenging, nuanced question that I can ask a number of LLMs to evaluate their intelligence. Respond only with the question."

question = "Please ask a question to assess someone's IQ. Respond only with the question."
messages = [{"role": "user", "content": question}]
# ask it using Azure OpenAI
response = client.chat.completions.create(
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    messages=messages
)

question = response.choices[0].message.content
print(question)

# -----------------------------
# Azure OpenAI Test
# -----------------------------


messages = [{"role": "user", "content": question}]


response = client.chat.completions.create(
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    messages=messages
)
azureOpenAIResponse = response.choices[0].message.content
competitors.append("OpenAI 4o-mini")
answers.append(azureOpenAIResponse)
# -----------------------------
#  Gemini Test
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
competitors.append("Gemini 2.5 Flash")
answers.append(geminiResponse)

# -----------------------------
# LLM as a Judge
# -----------------------------

# Iterate over two lists (competitors and answers) in parallel using zip().
# zip() pairs up elements from both lists by their position (index),
# so competitor[i] is matched with answer[i].
# This avoids manual indexing and makes the loop cleaner and more Pythonic.
for competitor, answer in zip(competitors, answers):
    print(f"Competitor: {competitor}\n\n{answer}")


# Let's bring this together - note the use of "enumerate"

combined = ""
for index, answer in enumerate(answers):
    combined += f"# Response from competitor {index+1}\n\n"
    combined += answer + "\n\n"



judge = f"""You are judging a competition between {len(competitors)} competitors.
Each model has been given this question:

{question}

Your job is to evaluate each response for clarity and strength of argument, and rank them in order of best to worst.
Respond with JSON, and only JSON, with the following format:
{{"results": [{{"name": "OpenAI 4o-mini"}}, {{"name": "Gemini 2.5 Flash"}}]}}

Here are the responses from each competitor:

{combined}

Now respond with the JSON with the ranked order of the competitors, nothing else. Do not include markdown formatting or code blocks."""# Initialize Ollama client using environment variables
ollama = OpenAI(
    base_url=os.getenv("OLLAMA_LOCAL_API_BASE"),
    api_key=os.getenv("OLLAMA_LOCAL_KEY")
)
judge_model_name = "deepseek-r1:1.5b"
judge_messages = [{"role": "user", "content": judge}]
response = ollama.chat.completions.create(
    model=judge_model_name, messages=judge_messages
    )
judge_evaluation_result = response.choices[0].message.content

print("Raw response:", judge_evaluation_result)
