## 01-langchain_setup.py

### What I will build?
- A minimal FastAPI endpoint using LangChain's `ChatOpenAI` with Gemini via an OpenAI-compatible API.

## WHat concepts gets applied in this exercise?
- Loading environment variables with `dotenv`
- Initializing LangChain LLMs for non-OpenAI providers using `base_url`
- Composing messages with `SystemMessage` and `HumanMessage`
- Parsing model responses with LangChain

### What skills I will acuire post completion of this exercise?
- Configure LangChain `ChatOpenAI` to talk to Gemini
- Structure prompts using system and user roles
- Handle basic API errors in FastAPI

### What tools, libraries, frameworks  I will use for this exercise?
- FastAPI, Uvicorn
- `langchain-openai` (LangChain integration), `langchain-core`
- `python-dotenv` for `.env`

### How to run this example?
```bash
uv run uvicorn 02-langchain_agents.01-langchain_basic.01-langchain_setup:app --reload 
```
Required env vars: `GEMINI_API_KEY`, `GEMINI_API_BASE` (e.g., `https://generativelanguage.googleapis.com/v1beta/openai/`), optional `GEMINI_MODEL_NAME` (default: `gemini-2.5-flash`).

### What Next?
- Try different system prompts and temperatures
- Return token usage/latency in the response
- Add request validation and simple logging


## 02-langchain_asynchronous_llm_call.py

### What I will build?
- An async FastAPI endpoint that calls the LLM using `await llm.ainvoke(...)` for non-blocking I/O.

## WHat concepts gets applied in this exercise?
- Async vs sync LLM calls in LangChain (`ainvoke` vs `invoke`)
- Event loop-friendly FastAPI handlers
- Brief inline comments explaining where/why the await happens

### What skills I will acuire post completion of this exercise?
- Implement asynchronous LLM calls with LangChain
- Keep endpoints responsive under concurrent load
- Handle async exceptions cleanly

### What tools, libraries, frameworks  I will use for this exercise?
- FastAPI, Uvicorn
- `langchain-openai`, `langchain-core`
- `python-dotenv`

### How to run this example?
```bash
uv run uvicorn 02-langchain_agents.01-langchain_basic.02-langchain_asynchronous_llm_call:app --reload 
```
Required env vars: same as above (`GEMINI_API_KEY`, `GEMINI_API_BASE`, optional `GEMINI_MODEL_NAME`).

### What Next?
- Compare performance between `invoke` and `ainvoke`
- Add streaming with `astream_events` for token-by-token responses
- Batch multiple requests concurrently with `asyncio.gather`


## 03-langchain_prompt_templates.py

### What I will build?
- A FastAPI endpoint where user input fills a `PromptTemplate` (e.g., "Summarize {topic} in 3 points").

## WHat concepts gets applied in this exercise?
- `PromptTemplate` for separating static and dynamic prompt parts

### What skills I will acuire post completion of this exercise?
- Design reusable prompts and vary inputs (topic, tone)
- Swap templates for different styles (formal vs casual)

### What tools, libraries, frameworks  I will use for this exercise?
- FastAPI, Uvicorn, LangChain
- `langchain`, `langchain-openai`, `langchain-core`

### How to run this example?
```bash
uv run uvicorn 02-langchain_agents.01-langchain_basic.03-langchain_prompt_templates:app --reload 
```
Required env vars: same as above (`GEMINI_API_KEY`, `GEMINI_API_BASE`, optional `GEMINI_MODEL_NAME`).

### What Next?
- Try multiple templates (formal vs casual tone)
- Add input validation for template variables
- Build a small library of reusable templates



## 04-langchain_output_parser.py

### What I will build?
- A FastAPI app exposing two endpoints that demonstrate LangChain output parsers:
  - `/parse_list` using `CommaSeparatedListOutputParser`
  - `/parse_json` using `JsonOutputParser`

### WHat concepts gets applied in this exercise?
- Parser-driven prompting with `get_format_instructions()`
- Structured outputs: list parsing and JSON parsing
- Using `PromptTemplate` to inject parser format instructions

### What skills I will acuire post completion of this exercise?
- Design prompts that reliably produce parseable outputs
- Parse model responses into Python types (list, dict)
- Handle parsing errors in FastAPI

### What tools, libraries, frameworks  I will use for this exercise?
- FastAPI, Uvicorn
- `langchain-openai`, `langchain-core`
- `python-dotenv`

### How to run this example?
```bash
uv run uvicorn 02-langchain_agents.01-langchain_basic.04-langchain_output_parser:app --reload
```
Required env vars: `GEMINI_API_KEY`, `GEMINI_API_BASE` (e.g., `https://generativelanguage.googleapis.com/v1beta/openai/`), optional `GEMINI_MODEL_NAME`.

### Example Requests
- `/parse_list`
  - Prompt: "List 5 popular programming languages as a comma separated list."
  - Parser: `CommaSeparatedListOutputParser`
  - Example Output: `["Python", "JavaScript", "Java", "C++", "Go"]`

- `/parse_json`
  - Prompt: "Return details of a programming language in JSON with fields: name, type (compiled/interpreted), and popularity (1-10)."
  - Parser: `JsonOutputParser`
  - Example Output:
    ```json
    {
      "name": "Python",
      "type": "interpreted",
      "popularity": 10
    }
    ```

### What Next?
- Add Zod or Pydantic schema-based parsing for stricter validation
- Extend `/parse_json` to return a list of N language objects
- Return token usage and latency metrics in responses

## 05-langchain_pydantic_output_parser.py

### What I will build?
- A FastAPI endpoint demonstrating `PydanticOutputParser` to coerce model output into a validated Pydantic model.

### WHat concepts gets applied in this exercise?
- Schema-first prompting with `PydanticOutputParser`
- Using `get_format_instructions()` to guide model output shape
- Validating and returning typed responses from FastAPI

### What skills I will acuire post completion of this exercise?
- Define Pydantic models for structured LLM output
- Prompt models to follow strict JSON structure
- Handle validation errors gracefully

### What tools, libraries, frameworks  I will use for this exercise?
- FastAPI, Uvicorn
- `langchain-openai`, `langchain-core`
- `python-dotenv`, `pydantic`

### How to run this example?
```bash
uv run uvicorn 02-langchain_agents.01-langchain_basic.05-langchain_pydantic_output_parser:app --reload
```
Required env vars: `GEMINI_API_KEY`, `GEMINI_API_BASE`, optional `GEMINI_MODEL_NAME`.

### Endpoint
- `/parse_pydantic` (GET, optional query `language`)
  - Model schema:
    - `name: str`
    - `type: "compiled" | "interpreted"`
    - `popularity: int` (1-10)
  - Example Output:
    ```json
    {
      "name": "Python",
      "type": "interpreted",
      "popularity": 10
    }
    ```

### What Next?
- Expand the schema with fields like `paradigms` (list) and `creator`
- Add input validation for `language` query and default behaviors
- Compare `JsonOutputParser` vs `PydanticOutputParser` error handling