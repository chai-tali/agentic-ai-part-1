# 🤖 Agentic AI Part 1 - LLM Apps & Testing

Educational repository for learning to build and test LLM applications with Python, FastAPI, and modern AI APIs.

## 🎯 What You'll Learn

- 🔌 **LLM Integration**: Connect to Azure OpenAI, Gemini, and local models
- 🚀 **FastAPI Development**: Build streaming and non-streaming API endpoints  
- 🧪 **Testing Patterns**: Unit, integration, and manual testing for AI apps
- ⚡ **Streaming Responses**: Real-time AI chat interfaces
- 🛠️ **Modern Tooling**: UV package management, pytest, async programming

## 🏃‍♂️ Quick Start

### 1. Setup Project
```bash
# Clone the repository
git clone https://github.com/ximanta/agentic-ai-part-1.git
cd agentic-ai-part-1

# Install dependencies (recommended)
uv venv
uv sync

# Alternative: Using pip
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -e ".[test]"
```

### 2. Configure Environment
```bash
# Copy example environment file
cp .envbackup .env

# Edit .env file with your API keys:
# - AZURE_OPENAI_API_KEY=your_azure_key
# - GEMINI_API_KEY=your_gemini_key
# - etc.
```

### 3. Start FastAPI Server
```bash
# Start the development server with auto-reload
uv run uvicorn 01-llm_apps.04-fast_api_llm_app:app --reload

# 🌐 Server runs at: http://localhost:8000
# 📚 API docs at: http://localhost:8000/docs
```

### 4. Run Tests
```bash
# Run all tests
uv run pytest tests

# Run with verbose output
uv run pytest tests -v

# Run specific test categories
uv run pytest tests/unit/ -v        # Unit tests only
uv run pytest tests/integration/ -v # Integration tests only
```

## 📁 Project Structure

```
agentic-ai-part-1/
├── 01-llm_apps/                    # Main application code
│   ├── 01-openai_sdk_foundation.py # Basic OpenAI SDK usage
│   ├── 02-local_model.py           # Local model integration
│   ├── 03-chat_streaming_response.py # Streaming concepts
│   └── 04-fast_api_llm_app.py      # FastAPI application
├── tests/                          # Comprehensive test suite
│   ├── unit/                       # Fast, isolated tests
│   ├── integration/                # End-to-end API tests
│   ├── manual/                     # Interactive testing
│   └── conftest.py                 # Test configuration
├── .env                           # Your API keys (create from .envbackup)
└── README.md                      # This file
```

## 🚀 Running the Examples

### Basic LLM Integration
```bash
# Basic OpenAI SDK usage
uv run python 01-llm_apps/01-openai_sdk_foundation.py

# Local model integration (requires Ollama)
uv run python 01-llm_apps/02-local_model.py

# Streaming response concepts
uv run python 01-llm_apps/03-chat_streaming_response.py
```

### FastAPI Application
```bash
# Start the API server
uv run uvicorn 01-llm_apps.04-fast_api_llm_app:app --reload

# Test endpoints manually
curl http://localhost:8000/health
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello AI!", "model": "azureopenai"}'
```

## 🧪 Testing Guide

### Test Categories

| Type | Purpose | Speed | Dependencies |
|------|---------|-------|--------------|
| **Unit** | Individual functions | Fast (< 1s) | Mocked APIs |
| **Integration** | End-to-end API | Medium (1-5s) | Real FastAPI |
| **Manual** | Interactive learning | Variable | Running server |

### Running Tests

```bash
# All tests
uv run pytest tests

# Specific test files
uv run pytest tests/unit/test_fastapi_functions.py -v
uv run pytest tests/integration/test_api_endpoints.py -v

# With coverage report
uv run pytest tests --cov=01-llm_apps --cov-report=html

# Manual testing (requires running server)
uv run python tests/manual/test_api_client.py
```

### Test Features Covered

- ✅ **Health Endpoints**: Server status checking
- ✅ **Chat Endpoints**: Non-streaming responses
- ✅ **Streaming Endpoints**: Real-time chat
- ✅ **Error Handling**: Authentication and validation errors
- ✅ **Request Validation**: Pydantic model testing
- ✅ **Mock Integration**: External API mocking

## 📚 Learning Path

### Beginner
1. Start with `01-openai_sdk_foundation.py` - Learn basic LLM integration
2. Run the FastAPI server and explore `/docs`
3. Run unit tests to understand isolated testing
4. Try manual testing with the interactive client

### Intermediate  
1. Study `04-fast_api_llm_app.py` - Understand streaming vs non-streaming
2. Examine integration tests for API testing patterns
3. Experiment with different models and parameters
4. Modify tests to add new functionality

### Advanced
1. Extend the API with new endpoints
2. Add new LLM providers (Anthropic, local models)
3. Implement authentication and rate limiting
4. Build a complete chat interface

## 🛠️ API Endpoints

| Endpoint | Method | Purpose | Parameters |
|----------|--------|---------|------------|
| `/health` | GET | Health check | None |
| `/chat` | POST | Non-streaming chat | `message`, `model` |
| `/chat/stream` | POST | Streaming chat | `message`, `model` |

### Model Options
- `azureopenai` - Azure OpenAI (GPT models)
- `gemini` - Google Gemini
- `both` - Both models (non-streaming only)

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Make sure you're in the project root
   cd agentic-ai-part-1
   uv run pytest tests
   ```

2. **Module Not Found**
   ```bash
   # Reinstall dependencies
   uv sync
   ```

3. **API Authentication Errors**
   - Check your `.env` file exists
   - Verify API keys are correct
   - Ensure you have API credits/quota

4. **Server Won't Start**
   ```bash
   # Check if port 8000 is available
   # Try a different port:
   uv run uvicorn 01-llm_apps.04-fast_api_llm_app:app --reload --port 8001
   ```

5. **Tests Failing**
   - Tests use mock API keys by default
   - Integration tests expect authentication errors with mock keys
   - Run `uv run pytest tests -v` for detailed output

### Getting Help

- 📖 Check the inline code comments and docstrings
- 🔍 Use `uv run pytest tests -v --tb=long` for detailed test output
- 📚 Explore `/docs` when the FastAPI server is running
- 💬 Read the learning notes in each test file

## 🎓 Educational Features

- **Comprehensive Comments**: Every function and test is documented
- **Progressive Complexity**: From basic scripts to full API testing
- **Real-World Patterns**: Production-ready code structure
- **Multiple Learning Styles**: Visual (API docs), hands-on (tests), reading (code)
- **Best Practices**: Modern Python, async/await, proper testing

## 📦 Dependencies

- **Core**: `fastapi`, `uvicorn`, `openai`, `python-dotenv`
- **Testing**: `pytest`, `pytest-asyncio`, `httpx`, `pytest-mock`
- **Development**: `uv` (package manager), `pydantic` (validation)

## 🚀 Next Steps

1. **Extend Functionality**: Add new LLM providers, authentication, rate limiting
2. **Build UIs**: Create web or mobile frontends for the API
3. **Deploy**: Learn containerization and cloud deployment
4. **Scale**: Implement caching, load balancing, monitoring

Happy Learning! 🎉