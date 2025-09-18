## 01-openai_sdk_foundation.py

### What I will build?
- A simple script that sends a prompt to Azure OpenAI (or compatible OpenAI API) and prints the response.

## WHat concepts gets applied in this exercise?
- Basic SDK setup and authentication via environment variables
- Chat completion request/response structure
- Error handling for API calls

### What skills I will acuire post completion of this exercise?
- Configure API clients securely with env vars
- Send a minimal chat request and parse the reply
- Troubleshoot common auth/config issues

### What tools, libraries, frameworks  I will use for this exercise?
- Python, `openai` SDK (AzureOpenAI client)
- `python-dotenv` for `.env`

### How to run this example?
```bash
uv run python 01-llm_apps/01-openai_sdk_foundation.py
```
Required env vars: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_API_VERSION`, `AZURE_OPENAI_DEPLOYMENT_NAME`.

### What Next?
- Try different prompts, system messages
- Capture token usage and latency metrics
- Add retries/backoff for transient errors


## 02-local_model.py

### What I will build?
- A script that talks to a local LLM via an OpenAI-compatible API (e.g., Ollama) to run offline-ish experiments.

## WHat concepts gets applied in this exercise?
- Swapping remote providers with local backends
- Using an OpenAI-compatible base URL and key
- Comparing response speed, quality, and cost trade-offs

### What skills I will acuire post completion of this exercise?
- Configure a local model endpoint
- Send requests to different backends with minimal code changes
- Assess quality/performance differences

### What tools, libraries, frameworks  I will use for this exercise?
- Python, `openai`-compatible client
- Local LLM runtime (e.g., Ollama)

### How to run this example?
```bash
uv run python 01-llm_apps/02-local_model.py
```
Ensure your local server is running (e.g., Ollama: `ollama serve`) and `OLLAMA_LOCAL_API_BASE`, `OLLAMA_LOCAL_KEY` are set if used.

### What Next?
- Test multiple local models and measure latency/quality
- Add streaming to compare UX
- Prompt-tune for better local results


## 03-chat_streaming_response.py

### What I will build?
- A small FastAPI app demonstrating non-streaming vs streaming chat responses.

## WHat concepts gets applied in this exercise?
- Server-Sent Events (SSE) for token streaming
- Differences between blocking and streaming API patterns
- FastAPI endpoint design

### What skills I will acuire post completion of this exercise?
- Implement and consume streaming responses
- Structure endpoints for different delivery modes
- Handle partial content safely

### What tools, libraries, frameworks  I will use for this exercise?
- FastAPI, Uvicorn, `openai`/`AzureOpenAI`

### How to run this example?
```bash
uv run uvicorn 01-llm_apps.03-chat_streaming_response:app --reload
# Visit http://localhost:8000 for endpoints and try /chat and /chat/stream
```

## 04-fast_api_llm_app.py

### What I will build?
- A FastAPI service exposing non-streaming and streaming chat endpoints, integrating Azure OpenAI and optionally Gemini.

## WHat concepts gets applied in this exercise?
- Multi-provider LLM integration
- Request/response models with Pydantic
- Error handling and health checks

### What skills I will acuire post completion of this exercise?
- Build production-style API endpoints
- Validate inputs and structure outputs
- Operate and debug a small LLM microservice

### What tools, libraries, frameworks  I will use for this exercise?
- FastAPI, Uvicorn, `openai`/`AzureOpenAI`

### How to run this example?
```bash
uv run uvicorn 01-llm_apps.04-fast_api_llm_app:app --reload
# Docs: http://localhost:8000/docs
```

## 05-override_default_hyperparameter.py

### What I will build?
- A script that demonstrates overriding model parameters (e.g., temperature, top_p) and observing output changes.

## WHat concepts gets applied in this exercise?
- Decoding strategies and their impact on output
- Parameter tuning to balance creativity vs consistency

### What skills I will acuire post completion of this exercise?
- Tune generation parameters intentionally
- Design quick experiments to compare outputs
- Document and reproduce parameter configurations

### What tools, libraries, frameworks  I will use for this exercise?
- Python, `openai`/`AzureOpenAI`

### How to run this example?
```bash
uv run python 01-llm_apps/05-override_default_hyperparameter.py
```

### What Next?
- Log outputs with metadata for later analysis
- Build a small parameter playground UI


## 06-tool_calling.py

### What I will build?
- A FastAPI app that uses LLM tool calling to fetch user location from IP and personalize responses.

## WHat concepts gets applied in this exercise?
- LLM tool/function calling schema and invocation
- Extracting client IP from requests; private IP detection and public-IP fallback
- Robust argument injection/validation for tools

### What skills I will acuire post completion of this exercise?
- Design and register tools for LLMs
- Safely pass runtime context (IP) into tool calls
- Handle third-party API errors and produce graceful fallbacks

### What tools, libraries, frameworks  I will use for this exercise?
- FastAPI, Uvicorn
- `openai`/`AzureOpenAI`
- `httpx` for async HTTP
- APIIP (`https://apiip.net/documentation`) for IP geolocation

### How to run this example?
```bash
uv run uvicorn 01-llm_apps.06-tool_calling:app --reload
# Open http://localhost:8000/docs and POST to /chat
# Example: {"message": "What is the history of my place?"}
```
Required env vars: `AZURE_OPENAI_*`, and `APIIP_KEY` (or `APIP_KEY` / `APIP_API_KEY`).
Notes: When using localhost, the app replaces private/loopback IPs with your public IP automatically.

### What Next?
- Add more tools (maps search, nearby POIs) and compose multi-tool flows
- Cache IP lookups to reduce latency and costs
- Log tool usage and surface tool results in responses for transparency


