import os
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory


# Load environment variables from .env file (works for direct and uvicorn runs)
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=str(env_path))


def initialize_gemini_llm() -> ChatOpenAI:
    """
    Initialize a LangChain ChatOpenAI client pointed at Gemini's
    OpenAI-compatible endpoint.

    Requires the following environment variables in .env:
      - GEMINI_API_KEY
      - GEMINI_API_BASE (e.g., https://generativelanguage.googleapis.com/v1beta/openai/)
      - GEMINI_MODEL_NAME (optional, defaults to "gemini-2.5-flash")
    """
    api_key = os.getenv("GEMINI_API_KEY")
    base_url = os.getenv("GEMINI_API_BASE")
    model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set. Please configure it in your .env file.")
    if not base_url:
        raise ValueError("GEMINI_API_BASE is not set. Please configure it in your .env file.")

    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
    )


app = FastAPI(title="LangChain + Gemini ConversationBufferMemory", version="1.0.0")
llm = initialize_gemini_llm()

# Memory persists for the lifetime of the server process
memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# Prompt with history placeholder
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a friendly chatbot that remembers the conversation."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

# Chain = LLM with memory
conversation_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
)


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    response: str
    history: List[str]


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Chat endpoint with memory.
    Memory persists across requests for this server run.
    """
    try:
        # apredict returns a string response
        result = await conversation_chain.apredict(input=request.query)
        # Flatten memory into strings for return
        history = [
            f"{msg.type}: {msg.content}" for msg in memory.chat_memory.messages
        ]
        return ChatResponse(response=result, history=history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8008)


