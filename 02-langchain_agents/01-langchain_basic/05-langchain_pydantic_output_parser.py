import os
from typing import Optional, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


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


app = FastAPI(title="LangChain + Gemini Pydantic Output Parser", version="1.0.0")
llm = initialize_gemini_llm()


class LanguageInfo(BaseModel):
    name: str
    type: Literal["compiled", "interpreted"]
    popularity: int = Field(ge=1, le=10)


@app.get("/parse_pydantic", response_model=LanguageInfo)
async def parse_pydantic(language: Optional[str] = None) -> LanguageInfo:
    """
    Generate and parse a structured object using PydanticOutputParser.

    Optional query param: `language` to suggest a specific language.
    """
    try:
        parser = PydanticOutputParser(pydantic_object=LanguageInfo)
        format_instructions = parser.get_format_instructions()

        prompt = PromptTemplate(
            template=(
                "Return details of a programming language in JSON with fields: "
                "name, type (compiled/interpreted), and popularity (1-10).\n"
                "{maybe_language}"
                "{format_instructions}"
            ),
            input_variables=["format_instructions", "maybe_language"],
        )

        maybe_language = "" if not language else f"Use the language: {language}.\n"
        final_prompt = prompt.format(
            format_instructions=format_instructions,
            maybe_language=maybe_language,
        )
        result = await llm.ainvoke(final_prompt)
        content = result.content if hasattr(result, "content") else str(result)
        # Token usage metadata
        print("Usage metadata:", result.response_metadata.get("token_usage", {}))
        return parser.parse(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parse error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


