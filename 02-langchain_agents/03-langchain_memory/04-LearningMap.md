## 01-langchain_buffer_memory.py

### What I will build?
- A conversational FastAPI endpoint that remembers previous messages across requests during the server run using `ConversationBufferMemory`.

### What concepts gets applied in this exercise?
- Stateless vs stateful LLM calls
- `ConversationBufferMemory` as the simplest memory
- How memory keeps track of history and injects it via `MessagesPlaceholder`

### What skills I will acquire post completion of this exercise?
- Manage conversational state with LangChain memory
- Design prompts that include a history placeholder
- Expose a chat API that returns both the model reply and the full history

### What tools, libraries, frameworks I will use for this exercise?
- FastAPI, Uvicorn
- `langchain-core`, `langchain-openai`, `langchain.chains`, `langchain.memory`
- `python-dotenv`

### How to run this example?
```bash
uv run uvicorn 02-langchain_agents.03-langchain_memory.01-langchain_buffer_memory:app --reload
```
Required env vars: `GEMINI_API_KEY`, `GEMINI_API_BASE` (e.g., `https://generativelanguage.googleapis.com/v1beta/openai/`), optional `GEMINI_MODEL_NAME`.

### Endpoint
- `POST /chat`
  - Request body:
    ```json
    { "query": "Hi, I am Pablo" }
    ```
  - Example interaction:
    - Request 1:
      ```json
      { "query": "Hi, I am Pablo" }
      ```
      Possible response:
      ```json
      {
        "response": "Hello Pablo! How are you today?",
        "history": [
          "human: Hi, I am Pablo",
          "ai: Hello Pablo! How are you today?"
        ]
      }
      ```
    - Request 2:
      ```json
      { "query": "What is my name?" }
      ```
      Possible response:
      ```json
      {
        "response": "You told me your name is Pablo.",
        "history": [
          "human: Hi, I am Pablo",
          "ai: Hello Pablo! How are you today?",
          "human: What is my name?",
          "ai: You told me your name is Pablo."
        ]
      }
      ```
  - Note: Memory is kept in a global variable for this server process, so all requests share the same conversation history until the server restarts.
\

## 02-langchain_window_memory.py

### What I will build?
- A conversational FastAPI endpoint that remembers only the last N turns using `ConversationBufferWindowMemory` with `k=2`.

### What concepts gets applied in this exercise?
- Why long context can be problematic (token cost, drift/hallucinations)
- How window size `k` limits remembered history
- Compare with full `ConversationBufferMemory`

### What skills I will acquire post completion of this exercise?
- Configure windowed memory to control context length
- Observe how older messages are “forgotten” by the chain
- Return both the latest response and current truncated history

### What tools, libraries, frameworks I will use for this exercise?
- FastAPI, Uvicorn
- `langchain-core`, `langchain-openai`, `langchain.chains`, `langchain.memory`
- `python-dotenv`

### How to run this example?
```bash
uv run uvicorn 02-langchain_agents.03-langchain_memory.02-langchain_window_memory:app --reload
```
Required env vars: `GEMINI_API_KEY`, `GEMINI_API_BASE` (e.g., `https://generativelanguage.googleapis.com/v1beta/openai/`), optional `GEMINI_MODEL_NAME`.

### Endpoint
- `POST /chat`
  - Example interaction (k=2):
    1. `{"query": "My name is Pablo"}` → "Hi Pablo!"
    2. `{"query": "I live in India"}` → "That's great, Pablo."
    3. `{"query": "What’s my name?"}` → Might forget, because only the last 2 exchanges are remembered.

## 03-langchain_summary_memory.py

### What I will build?
- A conversational FastAPI endpoint that keeps a rolling summary of past exchanges using `ConversationSummaryMemory`.

### What concepts gets applied in this exercise?
- Why summarization saves tokens in long conversations
- Memory compresses old turns into a concise summary
- Comparison: Buffer (everything) vs Window (last k) vs Summary (compressed all)

### What skills I will acquire post completion of this exercise?
- Configure `ConversationSummaryMemory` with the same LLM
- Inject `{history}` via LCEL and return the current summary
- Understand tradeoffs between memory strategies

### What tools, libraries, frameworks I will use for this exercise?
- FastAPI, Uvicorn
- `langchain-core`, `langchain-openai`, `langchain.chains`, `langchain.memory`
- `python-dotenv`

### How to run this example?
```bash
uv run uvicorn 02-langchain_agents.03-langchain_memory.03-langchain_summary_memory:app --reload
```
Required env vars: `GEMINI_API_KEY`, `GEMINI_API_BASE` (e.g., `https://generativelanguage.googleapis.com/v1beta/openai/`), optional `GEMINI_MODEL_NAME`.

### Endpoints
- `POST /chat` (runs the chain, updates summary, returns `response` and `summary`)
- `GET /memory` (inspect current memory variables)
- `POST /clear` (clear memory)

Example interaction:
1. `{"query": "My name is Pablo and I live in India."}` → response; summary may be short/empty initially.
2. `{"query": "I work as a trainer."}` → response; summary now includes prior details.
3. `{"query": "What’s my name and job?"}` → "You are Pablo, a trainer living in India." with a concise summary.

Note: Run multiple requests against the same server process (avoid auto-reload) so memory persists between calls.


## 04-langchain_hybrid_memory.py

### What I will build?
- A conversational FastAPI service that uses a custom summary-buffer memory to keep a fixed number of recent message pairs and summarize older ones. It includes endpoints to chat, inspect memory, view stats, and clear memory.

### What concepts gets applied in this exercise?
- Custom hybrid memory: recent detailed buffer + rolling summary of older turns
- Summarization trigger via `max_message_pairs` (older pairs summarized when the limit is exceeded)
- Injecting `{history}` with `MessagesPlaceholder` using LCEL
- LCEL chain assembly: `assign(history) -> prompt -> llm -> StrOutputParser`
- Inspecting and clearing in-process memory via API

### What skills I will acquire post completion of this exercise?
- Implement and use a custom memory wrapper over an LLM
- Expose endpoints to view summaries, recent messages, and raw memory
- Observe when/what gets summarized as the buffer exceeds its size
- Clear and reset memory state for fresh runs

### What tools, libraries, frameworks I will use for this exercise?
- FastAPI, Uvicorn
- `langchain-core`, `langchain-openai`, `langchain.memory`
- `python-dotenv`

### How to run this example?
```bash
uv run uvicorn 02-langchain_agents.03-langchain_memory.04-langchain_hybrid_memory:app --reload
```
Required env vars: `GEMINI_API_KEY`, `GEMINI_API_BASE` (e.g., `https://generativelanguage.googleapis.com/v1beta/openai/`), optional `GEMINI_MODEL_NAME` (default `gemini-2.5-flash`).

### Endpoints
- `POST /chat`
  - Body: `{ "query": "..." }`
  - Runs the chain with memory; returns model `response` and `memory_details` (summary, recent_message_pairs, truncated recent_messages, has_summary, max_message_pairs).
- `GET /memory/stats`
  - Returns `current_summary`, `recent_messages_count` (each pair = 2 messages), and a human-readable `memory_structure` description.
- `GET /memory/raw`
  - Returns `summary`, full `recent_messages`, `max_message_pairs`, and `memory_approach`.
- `POST /memory/clear`
  - Clears all stored memory for a fresh conversation state.
  

### Notes
- Default `max_message_pairs=3`. When exceeded, the two oldest pairs are summarized and appended to the running `summary`.
- `get_memory_variables()` builds `history` as: optional `AIMessage` with summary, followed by alternating `HumanMessage`/`AIMessage` for each recent pair.
- Summarization uses the same chat model via `llm.invoke` with a simple prompt; if it fails, it falls back to a truncated textual hint.
- This is a custom in-process implementation without token counting; behavior differs from LangChain's built-in `ConversationSummaryBufferMemory`.
- Memory is stored in-process; restarting the server resets it.