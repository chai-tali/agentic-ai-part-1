import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate


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
        temperature=0.4,
    )


app = FastAPI(title="LangChain + Gemini LLMChain", version="1.0.0")
llm = initialize_gemini_llm()


class BlogIdeaRequest(BaseModel):
    topic: str


class BlogIdeaResponse(BaseModel):
    idea: str
    model: str


# Define the prompt template for the LLMChain
prompt = PromptTemplate(
    template=(
        "You are a creative blog ideation assistant.\n"
        "Generate one short, catchy blog post idea for the topic: {topic}.\n"
        "Return a single sentence under 20 words."
    ),
    input_variables=["topic"],
)

# Compose the chain (LLMChain equivalent using LCEL)
chain = prompt | llm


@app.post("/blog_idea", response_model=BlogIdeaResponse)
async def blog_idea(request: BlogIdeaRequest) -> BlogIdeaResponse:
    """
    Minimal LLMChain: takes a user's topic and generates a short blog idea.
    """
    try:
        result = await chain.ainvoke({"topic": request.topic})
        content = result.content if hasattr(result, "content") else str(result)
        return BlogIdeaResponse(idea=content.strip(), model=llm.model_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8004)


