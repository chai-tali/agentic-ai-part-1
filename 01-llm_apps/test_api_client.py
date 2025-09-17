# FastAPI LLM Streaming API Test Client
# This script demonstrates how to test both streaming and non-streaming endpoints
# Perfect for learning how to consume LLM APIs programmatically

import requests  # For making HTTP requests
import json      # For parsing JSON responses
import time      # For delays and timing

# =============================================================================
# API TESTING CONFIGURATION
# =============================================================================

# Base URL of your FastAPI server
# Make sure your FastAPI app is running on this address!
BASE_URL = "http://localhost:8000"

# This client demonstrates:
# 1. Testing non-streaming endpoints (traditional API calls)
# 2. Testing streaming endpoints (real-time Server-Sent Events)
# 3. Error handling and response validation
# 4. Different request patterns for different endpoint types

# =============================================================================
# NON-STREAMING ENDPOINT TESTING
# =============================================================================

def test_chat_endpoint():
    """
    Test the NON-STREAMING /chat endpoint
    
    This demonstrates:
    - Traditional API request/response pattern
    - Sending JSON payload with message and model selection
    - Receiving complete responses from multiple models
    - Handling success/error responses
    """
    print("🧪 Testing NON-STREAMING /chat endpoint...")
    
    # Prepare request payload (updated format with simplified message structure)
    payload = {
        "message": "What is artificial intelligence?",  # Simple message string
        "model": "both"  # Test both models at once
    }
    
    print(f"📤 Sending request: {payload}")
    
    # Make the HTTP POST request
    # This will WAIT for the complete response (non-streaming behavior)
    response = requests.post(f"{BASE_URL}/chat", json=payload)
    
    # Handle the response
    if response.status_code == 200:
        data = response.json()
        print("✅ Non-streaming chat endpoint successful!")
        print(f"📝 Azure OpenAI Response: {data.get('azureopenai_response', 'None')[:100]}...")
        print(f"📝 Gemini Response: {data.get('gemini_response', 'None')[:100]}...")
        print(f"🎯 Model used: {data.get('model_used', 'Unknown')}")
    else:
        print(f"❌ Chat endpoint failed: {response.status_code}")
        print(f"Error details: {response.text}")

# =============================================================================
# STREAMING ENDPOINT TESTING
# =============================================================================

def test_streaming_endpoint():
    """
    Test the REAL-TIME STREAMING /chat/stream endpoint
    
    This demonstrates:
    - Server-Sent Events (SSE) consumption
    - Real-time text streaming
    - Processing chunks as they arrive
    - Handling streaming completion signals
    """
    print("\n🌊 Testing REAL-TIME STREAMING /chat/stream endpoint...")
    
    # Prepare streaming request payload (updated format)
    payload = {
        "message": "Explain machine learning in simple terms",  # Simple message string
        "model": "azureopenai"  # Single model for streaming
    }
    
    print(f"📤 Sending streaming request: {payload}")
    
    # Make streaming HTTP POST request
    # stream=True enables processing response as it arrives
    response = requests.post(
        f"{BASE_URL}/chat/stream",  # Updated endpoint (no model in URL)
        json=payload,
        stream=True  # This enables streaming on the client side
    )
    
    if response.status_code == 200:
        print("✅ Streaming endpoint successful!")
        print("🌊 Real-time streaming response:")
        print("-" * 40)
        
        # Process Server-Sent Events (SSE) stream
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                
                # SSE format: "data: {json_content}"
                if line_str.startswith('data: '):
                    data_str = line_str[6:]  # Remove 'data: ' prefix
                    
                    # Check for stream completion signal
                    if data_str == '[DONE]':
                        print("\n🏁 Stream completed!")
                        break
                    
                    # Parse and display each chunk
                    try:
                        data = json.loads(data_str)
                        content = data.get('content', '')
                        print(content, end='', flush=True)  # Real-time display
                    except json.JSONDecodeError:
                        # Skip malformed chunks
                        pass
        
        print("\n" + "-" * 40)
        print("✨ Streaming demonstration complete!")
    else:
        print(f"❌ Streaming endpoint failed: {response.status_code}")
        print(f"Error details: {response.text}")

# =============================================================================
# UTILITY ENDPOINT TESTING
# =============================================================================

def test_health_check():
    """
    Test the /health endpoint
    
    This demonstrates:
    - Simple GET request testing
    - API availability checking
    - Basic endpoint validation
    """
    print("\n🩺 Testing API health check...")
    
    # Simple GET request to health endpoint
    response = requests.get(f"{BASE_URL}/health")
    
    if response.status_code == 200:
        health_data = response.json()
        print("✅ API is healthy and ready!")
        print(f"📊 Health status: {health_data}")
    else:
        print(f"❌ Health check failed: {response.status_code}")
        print("Make sure your FastAPI server is running!")

# =============================================================================
# MAIN TESTING EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("🚀 FastAPI LLM Streaming API Test Suite")
    print("=" * 55)
    print("🎯 This script tests both streaming and non-streaming endpoints")
    print("🔍 Learn how to consume LLM APIs programmatically!")
    print("=" * 55)
    
    # Give server a moment to be ready
    print("⏳ Waiting for server to be ready...")
    time.sleep(2)
    
    try:
        # Test sequence: health check first, then functionality
        test_health_check()        # Verify API is running
        test_chat_endpoint()       # Test non-streaming endpoint
        test_streaming_endpoint()  # Test streaming endpoint
        
        print("\n" + "=" * 55)
        print("🎆 All tests completed successfully!")
        print("📚 You've learned how to:")
        print("   • Test non-streaming LLM APIs (traditional)")
        print("   • Consume real-time streaming responses (modern)")
        print("   • Handle different response formats")
        print("   • Process Server-Sent Events (SSE)")
        print("   • Build robust API clients with error handling")
        print("\n🚀 Ready to build your own LLM applications!")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ CONNECTION ERROR")
        print("Could not connect to the FastAPI server.")
        print("\n🔧 TROUBLESHOOTING:")
        print("1. Make sure your FastAPI server is running")
        print("2. Run: python 01-llm_apps/04-fast-api-llm-app.py")
        print("3. Check that it's accessible at http://localhost:8000")
        print("4. Verify your .env file has the correct API keys")
        
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        print("Check the server logs for more details.")

# =============================================================================
# LEARNING SUMMARY FOR API TESTING
# =============================================================================
"""
🎆 CONGRATULATIONS! You've learned API testing fundamentals!

KEY CONCEPTS DEMONSTRATED:

1. HTTP REQUEST PATTERNS:
   - GET requests for health checks and status
   - POST requests with JSON payloads for functionality
   - Different content types and headers

2. NON-STREAMING API TESTING:
   - Traditional request/response pattern
   - JSON payload construction
   - Response validation and parsing
   - Error handling for failed requests

3. STREAMING API TESTING:
   - Server-Sent Events (SSE) consumption
   - Real-time data processing
   - Stream completion detection
   - Chunk-by-chunk response handling

4. ROBUST CLIENT PATTERNS:
   - Connection error handling
   - Response validation
   - Status code checking
   - Graceful error messages

5. TESTING METHODOLOGY:
   - Health checks before functionality tests
   - Progressive testing (simple to complex)
   - Clear success/failure reporting
   - Helpful troubleshooting guidance

REAL-WORLD APPLICATIONS:
- Building LLM-powered applications
- API integration and testing
- Chat interface backends
- Streaming data processing
- Microservice communication

NEXT STEPS:
- Modify payloads to test different scenarios
- Add more sophisticated error handling
- Build a full chat interface using these patterns
- Explore different LLM providers and models
- Create automated test suites for your APIs
"""
