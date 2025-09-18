import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
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
# Custom Memory Implementation 
# ----------------------------
class CustomSummaryBufferMemory:
    def __init__(self, llm, max_message_pairs=5, memory_key="history"):
        self.llm = llm
        self.max_message_pairs = max_message_pairs
        self.memory_key = memory_key
        self.summary = ""
        self.recent_messages = []
    
    def add_message(self, user_message: str, ai_message: str):
        # Add new message pair
        self.recent_messages.append({"user": user_message, "ai": ai_message})
        
        # If we exceed the limit, summarize older messages
        if len(self.recent_messages) > self.max_message_pairs:
            # Take the oldest message pairs for summarization
            to_summarize = self.recent_messages[:2]  # Take 2 oldest pairs
            self.recent_messages = self.recent_messages[2:]  # Keep the rest
            
            # Create summary of old messages
            summary_text = self._create_summary(to_summarize)
            if self.summary:
                self.summary = f"{self.summary}\n\n{summary_text}"
            else:
                self.summary = summary_text
    
    def _create_summary(self, message_pairs):
        # Convert message pairs to text for summarization
        conversation_text = ""
        for pair in message_pairs:
            conversation_text += f"User: {pair['user']}\nAssistant: {pair['ai']}\n\n"
        
        # Create a simple summary prompt
        summary_prompt = f"""Please provide a concise summary of this conversation:

{conversation_text}

Summary:"""
        
        try:
            summary = self.llm.invoke(summary_prompt).content
            return f"Previous conversation summary: {summary}"
        except Exception:
            return f"Previous conversation included discussion about: {conversation_text[:200]}..."
    
    def get_memory_variables(self):
        messages = []
        
        # Add summary as a system message if it exists
        if self.summary:
            messages.append(AIMessage(content=f"Context from previous conversation: {self.summary}"))
        
        # Add recent messages
        for pair in self.recent_messages:
            messages.append(HumanMessage(content=pair["user"]))
            messages.append(AIMessage(content=pair["ai"]))
        
        return {self.memory_key: messages}
    
    def clear(self):
        self.summary = ""
        self.recent_messages = []

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="LangChain + Custom Summary Buffer Memory", version="1.0.0")
llm = initialize_gemini_llm()

# ----------------------------
# Custom Memory: Hybrid approach without token counting
# ----------------------------
custom_memory = CustomSummaryBufferMemory(
    llm=llm, 
    max_message_pairs=3,  # Keep 3 recent message pairs in full detail
    memory_key="history"
)

# ----------------------------
# Prompt: injects conversation history
# ----------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a friendly educational assistant that remembers conversation history. 
        You can recall details about the user from both recent messages and summarized older conversations.
        Always try to reference previous context when relevant."""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

# ----------------------------
# Create chain using LCEL
# ----------------------------
def get_memory_variables():
    return custom_memory.get_memory_variables()

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
    memory_details: Dict[str, Any]

class MemoryStats(BaseModel):
    current_summary: str
    recent_messages_count: int
    memory_structure: str

# ----------------------------
# Helper function to format memory details
# ----------------------------
def format_memory_details() -> Dict[str, Any]:
    """Format memory variables for educational purposes"""
    return {
        "summary": custom_memory.summary or "No summary yet",
        "recent_message_pairs": len(custom_memory.recent_messages),
        "recent_messages": [
            {
                "user": pair["user"][:100] + "..." if len(pair["user"]) > 100 else pair["user"],
                "ai": pair["ai"][:100] + "..." if len(pair["ai"]) > 100 else pair["ai"]
            }
            for pair in custom_memory.recent_messages
        ],
        "has_summary": bool(custom_memory.summary),
        "max_message_pairs": custom_memory.max_message_pairs
    }

# ----------------------------
# Chat endpoint
# ----------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        # Run the chain with input
        result_text = await chain.ainvoke({"input": request.query})
        
        # Save the conversation to custom memory
        custom_memory.add_message(request.query, result_text)
        
        return ChatResponse(
            response=result_text,
            memory_details=format_memory_details()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# ----------------------------
# Get detailed memory statistics
# ----------------------------
@app.get("/memory/stats", response_model=MemoryStats)
async def get_memory_stats():
    try:
        recent_count = len(custom_memory.recent_messages) * 2  # Each pair = 2 messages
        
        # Determine memory structure
        has_summary = bool(custom_memory.summary)
        structure = f"{'Summary + ' if has_summary else ''}{len(custom_memory.recent_messages)} recent message pairs"
        
        return MemoryStats(
            current_summary=custom_memory.summary or "No summary yet",
            recent_messages_count=recent_count,
            memory_structure=structure
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# ----------------------------
# Get raw memory for debugging
# ----------------------------
@app.get("/memory/raw")
async def get_raw_memory():
    try:
        return {
            "summary": custom_memory.summary,
            "recent_messages": custom_memory.recent_messages,
            "max_message_pairs": custom_memory.max_message_pairs,
            "memory_approach": "Custom implementation without token counting"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# ----------------------------
# Clear memory
# ----------------------------
@app.post("/memory/clear")
async def clear_memory():
    try:
        custom_memory.clear()
        return {"message": "Memory cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Run server
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)