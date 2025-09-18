import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableBranch


# Environment setup
# Load environment variables from a project-level .env so this script works
# both when executed directly (python file.py) and via uvicorn reload.
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


app = FastAPI(title="LangChain + Gemini Router Chain (LCEL)", version="1.0.0")
llm = initialize_gemini_llm()


class RouterRequest(BaseModel):
    query: str


class RouterResponse(BaseModel):
    output: str
    route: str
    model: str


# Parser
# Convert LLM message objects into plain strings so we can post-process
# and build uniform responses regardless of the provider's return type.
to_str = StrOutputParser()

# Blog idea chain (default branch)
# This prompt turns a free-form topic into a short, catchy blog idea.
prompt_blog = PromptTemplate(
    template=(
        "You are a creative blog ideation assistant.\n"
        "Generate one short, catchy blog post idea for the topic: {topic}.\n"
        "Return a single sentence under 20 words."
    ),
    input_variables=["topic"],
)

# RunnableLambda can adapt inputs into the shape a downstream component expects.
# Here the endpoint receives a payload with `query`, but our prompt expects `topic`.
map_query_to_topic = RunnableLambda(lambda x: {"topic": x["query"]})
blog_chain = (
    map_query_to_topic
    | prompt_blog
    | llm
    | to_str
    | RunnableLambda(lambda s: {"output": s.strip(), "route": "blog", "model": llm.model_name})
)


# Calculator function and chain (math branch)
# Educational note: eval() is DANGEROUS in general. We disable builtins and
# intend this only for very simple arithmetic. Do not use for untrusted code.
def safe_calculator(x: str) -> str:
    try:
        result = eval(x, {"__builtins__": {}})
        return f"Answer: {result}"
    except Exception:
        return "Sorry, I can only solve simple math problems."


math_chain = RunnableLambda(
    lambda x: {"output": safe_calculator(x["query"]), "route": "math", "model": "calculator"}
)


# Router chain using RunnableBranch
# RunnableBranch behaves like an if/else for chains.
# Condition: if the input text contains arithmetic operators, route to calculator.
# Otherwise: fall back to the blog idea generator.
router_chain = RunnableBranch(
    (lambda x: any(op in x["query"] for op in ["+", "-", "*", "/"]), math_chain),
    blog_chain,
)


@app.post("/route", response_model=RouterResponse)
async def route_query(request: RouterRequest) -> RouterResponse:
    """
    Router chain endpoint.

    Behavior:
    - If the query looks like a math expression (contains +, -, *, /),
      route to the calculator chain and return its answer.
    - Otherwise, route to the blog idea generator chain and return an idea.

    Example inputs:
    - {"query": "12 * 7 + 5"}  -> math
    - {"query": "AI in retail"} -> blog idea
    """
    try:
        result = await router_chain.ainvoke({"query": request.query})
        return RouterResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8007)


