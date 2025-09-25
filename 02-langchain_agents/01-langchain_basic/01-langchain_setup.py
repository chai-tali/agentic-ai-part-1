import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


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

    # ChatOpenAI supports overriding base_url to target OpenAI-compatible providers
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0.3,
    )


app = FastAPI(title="LangChain + Gemini Demo", version="1.0.0")
llm = initialize_gemini_llm()


class ChatRequest(BaseModel):
    
    system_prompt: Optional[str] = None
    message: str


class ChatResponse(BaseModel):
    response: str
    model: str


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Minimal chat endpoint using LangChain with Gemini via OpenAI-compatible API.
    """
    try:
        system_prompt = request.system_prompt or "You are a helpful assistant."
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=request.message),
        ]

        result = llm.invoke(messages)
        content = result.content if hasattr(result, "content") else str(result)
        # Token usage metadata
        print("Usage metadata:", result.response_metadata.get("token_usage", {}))
        return ChatResponse(response=content, model=llm.model_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


