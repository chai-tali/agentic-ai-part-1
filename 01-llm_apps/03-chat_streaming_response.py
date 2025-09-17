# Chat Streaming Response Tutorial
# This script demonstrates REAL-TIME STREAMING responses from LLMs
# Perfect for understanding how modern AI chat interfaces work (like ChatGPT)

import os
import json
from openai import OpenAI, AzureOpenAI
from IPython.display import Markdown, display
from dotenv import load_dotenv

# =============================================================================
# STREAMING CONCEPTS FOR BEGINNERS
# =============================================================================
"""
What is Streaming?
- Traditional API: Send request → Wait → Get complete response
- Streaming API: Send request → Get response chunks as they're generated
- User Experience: See text appearing word by word (like someone typing)

Why Use Streaming?
- Better user experience (no long waiting)
- Feels more interactive and responsive
- Users can start reading while AI is still generating
- Standard in modern AI chat applications

How It Works:
1. Add stream=True to your API call
2. Instead of one response, you get many small "chunks"
3. Each chunk contains a piece of the response
4. Display each chunk immediately as it arrives
"""

# Load environment variables from .env file
load_dotenv()
# =============================================================================
# PART 1: AZURE OPENAI STREAMING SETUP
# =============================================================================

# Initialize Azure OpenAI client (same as non-streaming setup)
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# The client setup is identical to non-streaming
# The magic happens in the API call parameters!

# =============================================================================
# USER INPUT AND MESSAGE FORMATTING
# =============================================================================

# Get user input (interactive)
# This makes the demo interactive - you can ask any question!
user_question = input("Enter your question (or 'quit' to exit): ")
if user_question.lower() == 'quit':
    print("Goodbye!")
    exit()

print(f"\nUser question: {user_question}")
print("=" * 50)

# Format message for OpenAI API
# Both streaming and non-streaming use the same message format
messages = [
    {
        "role": "user",        # Who is speaking ("user" or "assistant")
        "content": user_question  # What they said
    }
]

# This same message will be sent to both Azure OpenAI and Gemini
# So you can compare their responses to the same question!
# =============================================================================
# AZURE OPENAI STREAMING IN ACTION
# =============================================================================

print("Azure OpenAI Streaming Response:")
print("-" * 40)

# THE MAGIC HAPPENS HERE: stream=True enables real-time streaming!
stream = client.chat.completions.create(
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    messages=messages,
    stream=True  # 🔥 This single parameter enables streaming!
)

# Process the stream: each "chunk" contains a piece of the response
azure_response = ""  # We'll build the complete response as we go

for chunk in stream:  # Loop through each chunk as it arrives
    # Safety check: make sure the chunk has content
    # Not all chunks contain text (some are metadata)
    if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
        content = chunk.choices[0].delta.content  # Extract the text from this chunk
        
        # Display immediately (no buffering) - this creates the "typing" effect
        print(content, end="", flush=True)  # end="" prevents newlines, flush=True shows immediately
        
        # Also save it to build the complete response
        azure_response += content

print("\n" + "=" * 50)
print(f"\n📝 Azure OpenAI complete response saved ({len(azure_response)} characters)")

# =============================================================================
# PART 2: GEMINI STREAMING - Same Concept, Different Provider
# =============================================================================

# Initialize Gemini client using OpenAI SDK with different base_url
gemini_client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),     # Different API key
    base_url=os.getenv("GEMINI_API_BASE")    # Different endpoint URL
)

# Same streaming concept, different model!
print("\nGemini 2.5 Flash Streaming Response:")
print("-" * 40)

# Create streaming request (identical pattern to Azure OpenAI)
stream = gemini_client.chat.completions.create(
    model="gemini-2.5-flash",  # Gemini model name
    messages=messages,          # Same user question
    stream=True                 # Same streaming parameter!
)

# Process Gemini stream (identical logic to Azure OpenAI)
gemini_response = ""
for chunk in stream:
    # Same safety checks as Azure OpenAI
    if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
        content = chunk.choices[0].delta.content
        
        # Same immediate display logic
        print(content, end="", flush=True)
        
        # Same response building
        gemini_response += content

print("\n" + "=" * 50)
print(f"📝 Gemini complete response saved ({len(gemini_response)} characters)")

# =============================================================================
# STREAMING TUTORIAL SUMMARY
# =============================================================================
"""
🎆 CONGRATULATIONS! You just experienced real-time AI streaming!

KEY CONCEPTS YOU LEARNED:

1. STREAMING PARAMETER:
   - stream=True is the magic that enables real-time responses
   - Without it: wait for complete response (traditional)
   - With it: see text appear word by word (modern)

2. CHUNK PROCESSING:
   - Streaming returns many small "chunks" instead of one big response
   - Each chunk contains a piece of text (word, phrase, or sentence)
   - You process chunks immediately as they arrive

3. REAL-TIME DISPLAY:
   - print(content, end="", flush=True) shows text immediately
   - end="" prevents newlines between chunks
   - flush=True forces immediate display (no buffering)

4. RESPONSE BUILDING:
   - While displaying chunks, also save them to build complete response
   - Useful for logging, saving to database, or further processing

5. PROVIDER CONSISTENCY:
   - Same streaming pattern works with different LLM providers
   - Azure OpenAI and Gemini use identical streaming logic
   - Only difference is client initialization (API keys, endpoints)

6. USER EXPERIENCE:
   - Compare this to non-streaming - much more engaging!
   - Users see progress immediately
   - Feels like a conversation with a real person

TRY THIS:
- Run this script with different questions
- Notice how both models stream differently (speed, style)
- Compare the final responses - same question, different answers!
- Try very long questions to see streaming benefits

NEXT STEPS:
- Check out 04-fast-api-llm-app.py to see streaming in a web API
- Learn how to build streaming chat interfaces
- Experiment with different models and providers
"""
