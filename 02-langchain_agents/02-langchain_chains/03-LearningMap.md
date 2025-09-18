## 01-langchain_llm_chain_lcel.py

### What I will build?
- A minimal `LLMChain` that takes a user's topic and generates a short blog idea, exposed via a FastAPI endpoint.

### WHat concepts gets applied in this exercise?
- LLMChain basics (PromptTemplate + LLM + Chain using LCEL `prompt | llm`)
- Passing inputs into the chain
- Returning structured chain outputs from an API

### What skills I will acuire post completion of this exercise?
- Use `LLMChain` as the simplest composition in LangChain
- Design prompts with input variables and inject user-provided values
- Run chains inside FastAPI (async with `ainvoke`)

### What tools, libraries, frameworks  I will use for this exercise?
- FastAPI, Uvicorn
- `langchain-core`, `langchain-openai`
- `python-dotenv`

### How to run this example?
```bash
uv run uvicorn 02-langchain_agents.02-langchain_chains.01-langchain_llm_chain_lcel:app --reload
```
Required env vars: `GEMINI_API_KEY`, `GEMINI_API_BASE` (e.g., `https://generativelanguage.googleapis.com/v1beta/openai/`), optional `GEMINI_MODEL_NAME`.

### Endpoint
- `POST /blog_idea`
  - Request body:
    ```json
    { "topic": "AI in education" }
    ```
  - Example response:
    ```json
    {
      "idea": "Personalize learning with AI tutors for every student.",
      "model": "gemini-2.5-flash"
    }
    ```

## 02-langchain_llm_sequential_chain_lcel.py

### What I will build?
- A sequential LCEL pipeline that first generates a blog idea and then, using that idea, generates a concise outline.

### WHat concepts gets applied in this exercise?
- Chaining multiple steps with LCEL: `prompt1 | llm | prompt2 | llm`
- Passing intermediate structured outputs forward
- Reusing the same LLM across steps

### What skills I will acuire post completion of this exercise?
- Compose pipelines from smaller chains
- Capture intermediate results and return them alongside final outputs
- Run multi-step chains inside FastAPI (async with `ainvoke`)

### What tools, libraries, frameworks  I will use for this exercise?
- FastAPI, Uvicorn
- `langchain-core`, `langchain-openai`
- `python-dotenv`

### How to run this example?
```bash
uv run uvicorn 02-langchain_agents.02-langchain_chains.02-langchain_llm_sequential_chain_lcel:app --reload
```
Required env vars: `GEMINI_API_KEY`, `GEMINI_API_BASE` (e.g., `https://generativelanguage.googleapis.com/v1beta/openai/`), optional `GEMINI_MODEL_NAME`.

### Endpoint
- `POST /blog_plan`
  - Request body:
    ```json
    { "topic": "AI in healthcare" }
    ```
  - Example response:
    ```json
    {
      "idea": "AI triage assistants to speed up ER patient prioritization.",
      "outline": "1) Problem & impact 2) How triage AI works 3) Risks & ethics",
      "model": "gemini-2.5-flash"
    }
    ```

## 03-langchain_router_chain_lcel.py

### What I will build?
- A router chain that dynamically routes between a calculator (for simple math) and a blog idea generator (default branch).

### WHat concepts gets applied in this exercise?
- `RunnableBranch` for if-else style routing in LCEL
- Dynamic routing vs sequential chaining
- Mixing function tools with LLM chains

### What skills I will acuire post completion of this exercise?
- Detect patterns in input and route to the right chain
- Implement safe calculator logic alongside LLM prompts
- Return unified responses with route metadata

### What tools, libraries, frameworks  I will use for this exercise?
- FastAPI, Uvicorn
- `langchain-core`, `langchain-openai`
- `python-dotenv`

### How to run this example?
```bash
uv run uvicorn 02-langchain_agents.02-langchain_chains.03-langchain_router_chain_lcel:app --reload
```
Required env vars: `GEMINI_API_KEY`, `GEMINI_API_BASE` (e.g., `https://generativelanguage.googleapis.com/v1beta/openai/`), optional `GEMINI_MODEL_NAME`.

### Endpoint
- `POST /route`
  - Request body (math):
    ```json
    { "query": "12 * 7 + 5" }
    ```
  - Example response (math route):
    ```json
    {
      "output": "Answer: 89",
      "route": "math",
      "model": "calculator"
    }
    ```
  - Request body (blog):
    ```json
    { "query": "AI in retail" }
    ```
  - Example response (blog route):
    ```json
    {
      "output": "AI vision reduces checkout lines with smart carts.",
      "route": "blog",
      "model": "gemini-2.5-flash"
    }
    ```
