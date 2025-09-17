# Pytest Configuration and Fixtures for LLM Apps Testing
# This file provides shared test fixtures and configuration for all tests

import pytest
import os
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from typing import Generator

# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

# Configure pytest-asyncio
pytest_plugins = ["pytest_asyncio"]

# =============================================================================
# ENVIRONMENT FIXTURES - Mock environment variables for testing
# =============================================================================

@pytest.fixture
def mock_env_vars():
    """
    Mock environment variables for testing
    
    This fixture provides fake but valid-looking API keys and endpoints
    so tests don't need real credentials and won't make actual API calls
    """
    return {
        "AZURE_OPENAI_API_KEY": "test_azure_key_12345",
        "AZURE_OPENAI_ENDPOINT": "https://test-resource.openai.azure.com/",
        "AZURE_OPENAI_DEPLOYMENT_NAME": "test-gpt-4",
        "AZURE_OPENAI_API_VERSION": "2024-05-01-preview",
        "GEMINI_API_KEY": "test_gemini_key_67890",
        "GEMINI_API_BASE": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "OLLAMA_LOCAL_API_BASE": "http://localhost:11434/v1",
        "OLLAMA_LOCAL_KEY": "ollama"
    }

@pytest.fixture
def mock_environment(mock_env_vars, monkeypatch):
    """
    Apply mock environment variables to the test environment
    
    This fixture automatically sets up the environment for each test
    so they can run without real API keys
    """
    for key, value in mock_env_vars.items():
        monkeypatch.setenv(key, value)
    return mock_env_vars

# =============================================================================
# FASTAPI APP FIXTURES - For testing the web API
# =============================================================================

@pytest.fixture
def app(mock_environment):
    """
    FastAPI application fixture
    
    Returns the FastAPI app instance for testing
    Note: Import is done here to avoid import-time side effects
    """
    # Add the project root to Python path so we can import from 01-llm_apps
    import sys
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Now we can import the module using importlib
    import importlib.util
    spec = importlib.util.spec_from_file_location("fast_api_llm_app", 
                                                  os.path.join(project_root, "01-llm_apps", "04-fast_api_llm_app.py"))
    fast_api_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fast_api_module)
    
    return fast_api_module.app

@pytest.fixture
def client(app):
    """
    FastAPI test client fixture
    
    Provides a test client for making HTTP requests to the FastAPI app
    This is the recommended way to test FastAPI applications
    """
    from fastapi.testclient import TestClient
    return TestClient(app)

# =============================================================================
# MOCK RESPONSE FIXTURES - Simulate LLM API responses
# =============================================================================

@pytest.fixture
def mock_azure_response():
    """
    Mock Azure OpenAI API response
    
    Simulates what a real Azure OpenAI API would return
    """
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = "This is a test response from Azure OpenAI"
    return mock_response

@pytest.fixture
def mock_gemini_response():
    """
    Mock Gemini API response
    
    Simulates what a real Gemini API would return
    """
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = "This is a test response from Gemini"
    return mock_response

@pytest.fixture
def mock_streaming_chunks():
    """
    Mock streaming response chunks
    
    Simulates the chunks that come from a streaming LLM API
    Each chunk contains a small piece of the response
    """
    chunks = []
    words = ["Hello", " world", "!", " This", " is", " streaming", " text", "."]
    
    for word in words:
        chunk = Mock()
        chunk.choices = [Mock()]
        chunk.choices[0].delta = Mock()
        chunk.choices[0].delta.content = word
        chunks.append(chunk)
    
    # Add a final empty chunk (common in streaming APIs)
    final_chunk = Mock()
    final_chunk.choices = [Mock()]
    final_chunk.choices[0].delta = Mock()
    final_chunk.choices[0].delta.content = None
    chunks.append(final_chunk)
    
    return chunks

# =============================================================================
# CLIENT MOCK FIXTURES - Mock the actual LLM clients
# =============================================================================

@pytest.fixture
def mock_azure_client(mock_azure_response):
    """
    Mock Azure OpenAI client
    
    This replaces the real Azure OpenAI client with a mock
    so tests don't make real API calls
    """
    with patch('fast_api_llm_app.azure_client') as mock_client:
        mock_client.chat.completions.create.return_value = mock_azure_response
        yield mock_client

@pytest.fixture
def mock_gemini_client(mock_gemini_response):
    """
    Mock Gemini client
    
    This replaces the real Gemini client with a mock
    so tests don't make real API calls
    """
    with patch('fast_api_llm_app.gemini_client') as mock_client:
        mock_client.chat.completions.create.return_value = mock_gemini_response
        yield mock_client

@pytest.fixture
def mock_streaming_client(mock_streaming_chunks):
    """
    Mock streaming client
    
    This simulates a streaming response by returning an iterator of chunks
    """
    with patch('fast_api_llm_app.azure_client') as mock_client:
        mock_client.chat.completions.create.return_value = iter(mock_streaming_chunks)
        yield mock_client

# =============================================================================
# SAMPLE DATA FIXTURES - Common test data
# =============================================================================

@pytest.fixture
def sample_chat_request():
    """
    Sample chat request data for testing
    
    This provides realistic test data for API requests
    """
    return {
        "message": "What is artificial intelligence?",
        "model": "azureopenai"
    }

@pytest.fixture
def sample_chat_request_both_models():
    """
    Sample chat request for testing both models
    """
    return {
        "message": "Explain machine learning",
        "model": "both"
    }

@pytest.fixture
def sample_streaming_request():
    """
    Sample streaming request data
    """
    return {
        "message": "Write a short story about robots",
        "model": "azureopenai"
    }

# =============================================================================
# LEARNING NOTES FOR STUDENTS
# =============================================================================
"""
🎓 PYTEST FIXTURES EXPLAINED:

1. WHAT ARE FIXTURES?
   - Fixtures are reusable pieces of test setup
   - They provide data, mock objects, or configured environments
   - They run before your test functions

2. FIXTURE SCOPES:
   - function: Run once per test function (default)
   - class: Run once per test class
   - module: Run once per test file
   - session: Run once per test session

3. DEPENDENCY INJECTION:
   - Tests request fixtures by name as parameters
   - Pytest automatically provides the fixture values
   - Fixtures can depend on other fixtures

4. MOCKING BENEFITS:
   - Tests run faster (no real API calls)
   - Tests are reliable (no network dependencies)
   - Tests can simulate error conditions
   - Tests don't consume API quotas

5. COMMON PATTERNS:
   - mock_environment: Set up test environment variables
   - mock_clients: Replace external API clients
   - sample_data: Provide realistic test data
   - test_client: FastAPI testing client

TRY THIS:
- Add your own fixtures for specific test scenarios
- Experiment with different fixture scopes
- Create fixtures that simulate error conditions
- Build fixtures for different types of test data
"""
