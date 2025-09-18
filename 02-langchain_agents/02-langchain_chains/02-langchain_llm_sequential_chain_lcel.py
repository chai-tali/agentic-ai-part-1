import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda


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


app = FastAPI(title="LangChain + Gemini Sequential Chain (LCEL)", version="1.0.0")
llm = initialize_gemini_llm()


class BlogPlanRequest(BaseModel):
    topic: str


class BlogPlanResponse(BaseModel):
    idea: str
    outline: str
    model: str


# Step 1: Blog idea prompt
prompt_idea = PromptTemplate(
    template=(
        "You are a creative blog ideation assistant.\n"
        "Generate one short, catchy blog post idea for the topic: {topic}.\n"
        "Return a single sentence under 20 words."
    ),
    input_variables=["topic"],
)

# Step 2: Outline prompt consumes the idea from step 1
prompt_outline = PromptTemplate(
    template=(
        "Using the blog idea below, write a concise 3-bullet outline.\n"
        "Each bullet should be a single sentence.\n"
        "Idea: {idea}"
    ),
    input_variables=["idea"],
)

# Parsers
to_str = StrOutputParser()

# Sequential chain (demonstrates prompt1 | llm | prompt2 | llm)
sequential_chain = (
    prompt_idea
    | llm
    | to_str
    | RunnableLambda(lambda idea: {"idea": idea})
    | prompt_outline
    | llm
    | to_str
)


@app.post("/blog_plan", response_model=BlogPlanResponse)
async def blog_plan(request: BlogPlanRequest) -> BlogPlanResponse:
    """
    Step 1: Generate a blog idea from the topic.
    Step 2: Feed the idea into a second prompt to produce an outline.

    Demonstrates chaining with LCEL: prompt1 | llm | prompt2 | llm
    """
    try:
        # Compute idea (explicitly captured to return alongside outline)
        idea_chain = prompt_idea | llm | to_str
        idea = await idea_chain.ainvoke({"topic": request.topic})

        # Produce outline via the sequential LCEL chain (recomputes internally for demo clarity)
        outline = await sequential_chain.ainvoke({"topic": request.topic})

        return BlogPlanResponse(
            idea=idea.strip(),
            outline=outline.strip(),
            model=llm.model_name,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


