import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationSummaryMemory
from langchain_core.messages import HumanMessage, AIMessage

# ----------------------------
# Load environment variables
# ----------------------------
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=str(env_path))

# ----------------------------
# Initialize LLM (Gemini/OpenAI)
# ----------------------------
def initialize_gemini_llm() -> ChatOpenAI:
    api_key = os.getenv("GEMINI_API_KEY")
    base_url = os.getenv("GEMINI_API_BASE")
    model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")

    if not api_key or not base_url:
        raise ValueError("GEMINI_API_KEY and GEMINI_API_BASE must be set in .env")

    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0.5,
    )

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="LangChain + Gemini ConversationSummaryMemory", version="1.0.0")
llm = initialize_gemini_llm()

# ----------------------------
# Memory: summarizes conversation
# ----------------------------
summary_memory = ConversationSummaryMemory(
    llm=llm,
    memory_key="history",
    return_messages=True,
)

# ----------------------------
# Prompt: injects conversation history
# ----------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a friendly assistant that remembers the conversation history and can recall details about the user."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

# ----------------------------
# Create chain using LCEL (LangChain Expression Language)
# ----------------------------
def get_memory_variables():
    return summary_memory.load_memory_variables({})

chain = (
    RunnablePassthrough.assign(
        history=lambda x: get_memory_variables()["history"]
    )
    | prompt
    | llm
    | StrOutputParser()
)

# ----------------------------
# Request / Response models
# ----------------------------
class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str
    summary: str

# ----------------------------
# Chat endpoint
# ----------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        # Get current memory state
        memory_vars = summary_memory.load_memory_variables({})
        
        # Run the chain with input
        result_text = await chain.ainvoke({"input": request.query})
        
        # Save the conversation to memory
        summary_memory.save_context(
            {"input": request.query}, 
            {"output": result_text}
        )
        
        # Get updated memory state for summary
        updated_memory_vars = summary_memory.load_memory_variables({})
        history_messages = updated_memory_vars.get("history", [])
        
        # Create a readable summary
        if history_messages:
            if isinstance(history_messages[0], str):
                # If it's a string summary
                summary_text = history_messages[0]
            else:
                # If it's a list of messages
                summary_text = " | ".join([
                    f"{msg.type}: {msg.content}" for msg in history_messages
                ])
        else:
            summary_text = "No conversation history yet."

        return ChatResponse(
            response=result_text,
            summary=summary_text,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# ----------------------------
# Additional endpoint to view current memory
# ----------------------------
@app.get("/memory")
async def get_memory():
    try:
        memory_vars = summary_memory.load_memory_variables({})
        return {"memory": memory_vars}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# ----------------------------
# Endpoint to clear memory
# ----------------------------
@app.post("/clear")
async def clear_memory():
    try:
        summary_memory.clear()
        return {"message": "Memory cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# ----------------------------
# Run server
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)