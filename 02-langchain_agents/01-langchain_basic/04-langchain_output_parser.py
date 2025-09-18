import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import (
    CommaSeparatedListOutputParser,
    JsonOutputParser,
)


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


app = FastAPI(title="LangChain + Gemini Output Parsers", version="1.0.0")
llm = initialize_gemini_llm()

@app.get("/parse_list")
async def parse_list() -> list[str]:
    """
    Generate and parse a comma-separated list using CommaSeparatedListOutputParser.

    Example output: ["Python", "JavaScript", "Java", "C++", "Go"]
    """
    try:
        parser = CommaSeparatedListOutputParser()
        format_instructions = parser.get_format_instructions()

        prompt = PromptTemplate(
            template=(
                "List 5 popular programming languages as a comma separated list.\n"
                "{format_instructions}"
            ),
            input_variables=["format_instructions"],
        )

        final_prompt = prompt.format(format_instructions=format_instructions)
        result = await llm.ainvoke(final_prompt)
        content = result.content if hasattr(result, "content") else str(result)
        return parser.parse(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parse error: {str(e)}")


@app.get("/parse_json")
async def parse_json() -> dict:
    """
    Generate and parse JSON using JsonOutputParser.

    Expected fields: name, type (compiled/interpreted), popularity (1-10)
    """
    try:
        parser = JsonOutputParser()
        format_instructions = parser.get_format_instructions()

        prompt = PromptTemplate(
            template=(
                "Return details of a programming language in JSON with fields: "
                "name, type (compiled/interpreted), and popularity (1-10).\n"
                "{format_instructions}"
            ),
            input_variables=["format_instructions"],
        )

        final_prompt = prompt.format(format_instructions=format_instructions)
        result = await llm.ainvoke(final_prompt)
        content = result.content if hasattr(result, "content") else str(result)
        return parser.parse(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parse error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


