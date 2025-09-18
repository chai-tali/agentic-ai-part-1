# OpenAI SDK Foundation Tutorial
# This script demonstrates the basics of using OpenAI SDK with different LLM providers
# Perfect for beginners to understand how to connect to and use different AI models

import os
from openai import OpenAI, AzureOpenAI  # OpenAI SDK works with multiple providers
from IPython.display import Markdown, display
from dotenv import load_dotenv  # For loading API keys securely
from pathlib import Path

# =============================================================================
# ENVIRONMENT SETUP - Loading API Keys and Configuration
# =============================================================================

# Load environment variables from .env file
# Handle both direct execution and uvicorn execution
current_dir = Path(__file__).parent
project_root = current_dir.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=str(env_path))

# Your .env file should contain:
# AZURE_OPENAI_API_KEY=your_azure_key_here
# AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
# AZURE_OPENAI_DEPLOYMENT_NAME=your_model_deployment_name
# AZURE_OPENAI_API_VERSION=2024-05-01-preview
# GEMINI_API_KEY=your_gemini_key_here
# GEMINI_API_BASE=https://generativelanguage.googleapis.com/v1beta/openai/
# =============================================================================
# PART 1: AZURE OPENAI - Microsoft's OpenAI Service
# =============================================================================

# Initialize Azure OpenAI client
# Azure OpenAI is Microsoft's hosted version of OpenAI models
# Requires: API key, endpoint URL, API version, and deployment name
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),        # Your Azure OpenAI API key
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"), # API version (e.g., "2024-05-01-preview")
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")  # Your Azure resource endpoint
)

# Note: Azure OpenAI requires a "deployment" - you deploy a specific model
# (like GPT-4) to your Azure resource and give it a deployment name

# Step 1: Ask Azure OpenAI to generate an IQ question
# This demonstrates basic prompt engineering - asking for a specific type of output
question = "Please ask a question to assess someone's IQ. Respond only with the question."
messages = [{"role": "user", "content": question}]  # OpenAI chat format: list of message objects

# Make the API call to Azure OpenAI
# This is a NON-STREAMING request - we wait for the complete response
response = client.chat.completions.create(
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),  # Your deployed model name
    messages=messages  # The conversation history
)

# Extract the generated question from the response
question = response.choices[0].message.content
print(f"Generated IQ Question: {question}")

# Step 2: Now ask Azure OpenAI to answer its own question
# This demonstrates conversation flow and reusing generated content
messages = [{"role": "user", "content": question}]  # New conversation with the generated question

# Make another API call with the generated question
response = client.chat.completions.create(
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    messages=messages
)

# Get Azure OpenAI's answer to the IQ question
azureOpenAIResponse = response.choices[0].message.content
print(f"Azure OpenAI 4o-mini Answer: {azureOpenAIResponse}")

# =============================================================================
# PART 2: GEMINI - Google's AI Model via OpenAI-Compatible API
# =============================================================================

# Initialize Gemini client using OpenAI SDK
# Gemini can be accessed through an OpenAI-compatible API
# This means we use the same OpenAI SDK, just with different base_url
client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),    # Your Gemini API key from Google AI Studio
    base_url=os.getenv("GEMINI_API_BASE")   # Gemini's OpenAI-compatible endpoint
)

# Ask Gemini the same IQ question that Azure OpenAI generated
# This demonstrates how different models can give different answers to the same question
response = client.chat.completions.create(
    model="gemini-2.5-flash",  # Gemini model name (not a deployment like Azure)
    messages=[{"role": "user", "content": question}]  # Same question from Azure OpenAI
)

# Get Gemini's answer
geminiResponse = response.choices[0].message.content
print(f"Gemini 2.5 Flash Answer: {geminiResponse}")

# =============================================================================
# LEARNING SUMMARY
# =============================================================================
"""
KEY CONCEPTS DEMONSTRATED:

1. MULTIPLE LLM PROVIDERS:
   - Azure OpenAI: Microsoft's hosted OpenAI models
   - Gemini: Google's AI models via OpenAI-compatible API
   - Same SDK (OpenAI) works with different providers!

2. API PATTERNS:
   - Both use the same chat.completions.create() method
   - Same message format: [{"role": "user", "content": "..."}]
   - Different configuration (Azure needs deployment, Gemini needs base_url)

3. CONVERSATION FLOW:
   - Generate content with one model
   - Use that content as input for another request
   - Compare responses from different models

4. SECURITY BEST PRACTICES:
   - API keys stored in .env file (never in code!)
   - Use environment variables for configuration

5. RESPONSE STRUCTURE:
   - response.choices[0].message.content contains the AI's response
   - Consistent across different providers

TRY THIS:
- Run this script multiple times - you'll get different questions and answers!
- Compare how Azure OpenAI vs Gemini responds to the same question
- Experiment with different prompts in the 'question' variable
"""