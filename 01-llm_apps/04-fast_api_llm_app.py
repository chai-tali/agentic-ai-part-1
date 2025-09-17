# FastAPI LLM Streaming Tutorial
# This code demonstrates the difference between streaming and non-streaming responses

import os
import json
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse  # Special response type for streaming
from pydantic import BaseModel
from openai import OpenAI, AzureOpenAI
from dotenv import load_dotenv
import asyncio
import io

# Load environment variables from .env file (contains API keys and endpoints)
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="LLM Streaming API", version="1.0.0")

# =============================================================================
# PYDANTIC MODELS - Define the structure of requests and responses
# =============================================================================

class ChatMessage(BaseModel):
    """Individual chat message with role (user/assistant) and content"""
    role: str
    content: str

class ChatRequest(BaseModel):
    """Request format for both streaming and non-streaming endpoints"""
    message: str  # The user's question/prompt
    model: Optional[str] = "azureopenai"  # Which LLM to use
    # Options: "azureopenai", "gemini", or "both" (both only works for non-streaming)

class ChatResponse(BaseModel):
    """Response format for non-streaming endpoint (complete responses)"""
    azureopenai_response: Optional[str] = None  # Complete response from Azure OpenAI
    gemini_response: Optional[str] = None       # Complete response from Gemini
    model_used: str                             # Which model(s) were used

# Initialize clients with validation
def initialize_clients():
    # Check Azure OpenAI environment variables
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    

    
    if not all([azure_endpoint, azure_api_key, azure_api_version, azure_deployment]):
        print("WARNING: Missing Azure OpenAI environment variables")
    
    azure_client = AzureOpenAI(
        api_key=azure_api_key,
        api_version=azure_api_version,
        azure_endpoint=azure_endpoint
    )
    
    # Check Gemini environment variables
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    gemini_base_url = os.getenv("GEMINI_API_BASE")
    
    print(f"Gemini API Key: {'***' + gemini_api_key[-4:] if gemini_api_key else 'Not set'}")
    print(f"Gemini Base URL: {gemini_base_url}")
    
    gemini_client = OpenAI(
        api_key=gemini_api_key,
        base_url=gemini_base_url
    )
    
    return azure_client, gemini_client

azure_client, gemini_client = initialize_clients()

# =============================================================================
# NON-STREAMING FUNCTIONS - Wait for complete response before returning
# =============================================================================

def get_azureopenai_response(message: str) -> str:
    """
    NON-STREAMING: Get complete response from Azure OpenAI
    
    How it works:
    1. Send the entire message to Azure OpenAI
    2. WAIT for the complete response (blocking operation)
    3. Return the full text all at once
    
    User Experience: Traditional API call - you wait, then get everything
    """
    try:
        # Validate environment variables
        deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        if not deployment_name:
            raise ValueError("AZURE_OPENAI_DEPLOYMENT_NAME environment variable is not set")
        
        # Format message for OpenAI API
        messages = [{"role": "user", "content": message}]
        
        # KEY DIFFERENCE: NO stream=True parameter
        # This means we wait for the complete response
        response = azure_client.chat.completions.create(
            model=deployment_name,
            messages=messages
            # Notice: NO stream=True - this waits for complete response
        )
        
        # Return the complete response text
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Azure OpenAI Error: {str(e)}")
        raise e

def get_gemini_response(message: str) -> str:
    """
    NON-STREAMING: Get complete response from Gemini
    
    Same concept as Azure OpenAI - wait for complete response
    """
    # Format message for Gemini API (same format as OpenAI)
    messages = [{"role": "user", "content": message}]
    
    # Again: NO stream=True - wait for complete response
    response = gemini_client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=messages
        # No stream=True - waits for complete response
    )
    
    # Return complete response
    return response.choices[0].message.content

# =============================================================================
# STREAMING FUNCTION - Returns chunks of text as they're generated
# =============================================================================

def generate_streaming_response(message: str, model: str):
    """
    REAL-TIME STREAMING: Generate response chunks as they arrive
    
    How streaming works:
    1. Send message to LLM with stream=True
    2. LLM starts generating response immediately
    3. As each word/token is generated, we get a "chunk"
    4. We yield each chunk immediately (don't wait for complete response)
    5. User sees text appearing word by word in real-time
    
    This is a Python generator function (uses 'yield' instead of 'return')
    """
    # Format message for API
    messages = [{"role": "user", "content": message}]
    
    # Choose which LLM to stream from
    if model == "azureopenai":
        # KEY DIFFERENCE: stream=True enables real-time streaming
        stream = azure_client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            messages=messages,
            stream=True  # 🔥 This enables streaming!
        )
    elif model == "gemini":
        # Same for Gemini - stream=True enables streaming
        stream = gemini_client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=messages,
            stream=True  # 🔥 This enables streaming!
        )
    else:
        raise ValueError("Invalid model specified")
    
    # Process each chunk as it arrives (real-time)
    for chunk in stream:
        # Safety check: make sure chunk has content
        if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
            # Extract the text content from this chunk
            content = chunk.choices[0].delta.content
            
            # Yield this chunk immediately (don't wait for more)
            # Format as Server-Sent Events (SSE) for web browsers
            yield f"data: {json.dumps({'content': content})}\n\n"
    
    # Signal that streaming is complete
    yield "data: [DONE]\n\n"


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    NON-STREAMING ENDPOINT
    
    What happens:
    1. Receives your message
    2. Sends it to LLM(s) and WAITS for complete response
    3. Returns the full response(s) all at once
    
    User Experience: 
    - You wait (might take 5-30 seconds for long responses)
    - Then you get the complete answer instantly
    - Like traditional API calls
    
    Supports: "azureopenai", "gemini", or "both" models
    """
    try:
        # Initialize response variables
        azureopenai_response = None
        gemini_response = None
        
        # Call Azure OpenAI if requested
        if request.model in ["azureopenai", "both"]:
            print(f"Getting NON-STREAMING response from Azure OpenAI...")
            azureopenai_response = get_azureopenai_response(request.message)
            print(f"Azure OpenAI response complete: {len(azureopenai_response)} characters")
        
        # Call Gemini if requested
        if request.model in ["gemini", "both"]:
            print(f"Getting NON-STREAMING response from Gemini...")
            gemini_response = get_gemini_response(request.message)
            print(f"Gemini response complete: {len(gemini_response)} characters")
        
        # Return complete responses as JSON
        return ChatResponse(
            azureopenai_response=azureopenai_response,
            gemini_response=gemini_response,
            model_used=request.model
        )
    
    except Exception as e:
        print(f"Chat endpoint error: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    REAL-TIME STREAMING ENDPOINT
    
    What happens:
    1. Receives your message
    2. Starts sending it to the LLM with streaming enabled
    3. As the LLM generates each word/token, sends it immediately
    4. You see text appearing word by word in real-time
    
    User Experience:
    - Response starts appearing immediately (within 1-2 seconds)
    - Text appears progressively, word by word
    - Like watching someone type in real-time
    - Like ChatGPT, Claude, or other modern AI interfaces
    
    Technical Details:
    - Uses Server-Sent Events (SSE) for real-time communication
    - Only supports single models ("azureopenai" or "gemini")
    - Cannot do "both" because you can't mix two streams easily
    """
    # Validate model choice (streaming only supports single models)
    if request.model not in ["azureopenai", "gemini"]:
        raise HTTPException(
            status_code=400, 
            detail="Streaming only supports single models: 'azureopenai' or 'gemini' (no 'both')"
        )
    
    try:
        print(f"Starting STREAMING response from {request.model}...")
        
        # Return a StreamingResponse (special FastAPI response type)
        return StreamingResponse(
            generate_streaming_response(request.message, request.model),  # Generator function
            media_type="text/plain",
            headers={
                # Headers required for Server-Sent Events (SSE)
                "Cache-Control": "no-cache",      # Don't cache streaming responses
                "Connection": "keep-alive",       # Keep connection open for streaming
                "Content-Type": "text/event-stream"  # SSE format
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "streaming_ready": True}

# =============================================================================
# SUMMARY FOR LEARNERS: STREAMING vs NON-STREAMING
# =============================================================================
"""
KEY CONCEPTS FOR BEGINNERS:

1. NON-STREAMING (/chat):
   - Traditional API behavior
   - Send request → Wait → Get complete response
   - Uses: response = client.create(model=..., messages=...)  # NO stream=True
   - Good for: Short responses, when you need the complete text at once
   - User sees: Loading... then complete response appears instantly

2. STREAMING (/chat/stream):
   - Modern AI chat behavior (like ChatGPT)
   - Send request → Get chunks as they're generated → See text appear word by word
   - Uses: stream = client.create(model=..., messages=..., stream=True)  # stream=True!
   - Good for: Long responses, better user experience, feels more interactive
   - User sees: Text appearing progressively, word by word

3. TECHNICAL DIFFERENCES:
   - Non-streaming: Returns JSON with complete text
   - Streaming: Returns Server-Sent Events (SSE) with text chunks
   - Non-streaming: Can easily combine multiple models ("both")
   - Streaming: Only single model (harder to mix two streams)

4. WHEN TO USE WHICH:
   - Non-streaming: APIs, batch processing, when you need complete response
   - Streaming: Chat interfaces, long content generation, better UX

5. CODE PATTERNS:
   Non-streaming:  response = client.create(...); return response.choices[0].message.content
   Streaming:      stream = client.create(..., stream=True); for chunk in stream: yield chunk

TRY BOTH ENDPOINTS IN POSTMAN TO SEE THE DIFFERENCE!
"""

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting LLM Streaming Tutorial API...")
    print("📚 This demo shows the difference between streaming and non-streaming responses")
    print("🌐 Open http://localhost:8000 to see available endpoints")
    print("🧪 Test with Postman or curl to see streaming in action!\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
