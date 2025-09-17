import os
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Azure OpenAI Hyperparameter Demo", version="1.0.0")

class HyperparameterRequest(BaseModel):
    message: str
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 1000

class HyperparameterResponse(BaseModel):
    response: str
    parameters_used: dict
    response_length: int
    token_count: int

def initialize_azure_client():
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    
    if not all([azure_endpoint, azure_api_key, azure_api_version, azure_deployment]):
        raise ValueError("Missing required Azure OpenAI environment variables. Please check your .env file.")
    
    azure_client = AzureOpenAI(
        api_key=azure_api_key,
        api_version=azure_api_version,
        azure_endpoint=azure_endpoint
    )
    
    logger.info("Azure OpenAI client initialized successfully")
    logger.info(f"Endpoint: {azure_endpoint}")
    logger.info(f"Deployment: {azure_deployment}")
    
    return azure_client

azure_client = initialize_azure_client()

def get_azure_response_with_hyperparameters(
    message: str, 
    temperature: float = 1.0, 
    top_p: float = 0.9, 
    max_tokens: int = 1000
) -> dict:
    try:
        deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        if not deployment_name:
            raise ValueError("AZURE_OPENAI_DEPLOYMENT_NAME environment variable is not set")
        
        messages = [{"role": "user", "content": message}]
        
        logger.info(f"Using hyperparameters - Temperature: {temperature}, Top-p: {top_p}, Max tokens: {max_tokens}")
        
        response = azure_client.chat.completions.create(
            model=deployment_name,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )
        
        response_text = response.choices[0].message.content
        # Fnne for now. Going ahead use proper tokenizer (e.g., tiktoken).
        approximate_tokens = len(response_text) // 4
        
        return {
            "response": response_text,
            "parameters_used": {
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens
            },
            "response_length": len(response_text),
            "token_count": approximate_tokens
        }
        
    except Exception as e:
        logger.error(f"Azure OpenAI Error: {str(e)}")
        raise e

@app.post("/chat/hyperparameters", response_model=HyperparameterResponse)
async def chat_with_hyperparameters(request: HyperparameterRequest):
    try:
        logger.info(f"Request received - Message: {request.message[:50]}...")
        logger.info(f"Parameters - Temperature: {request.temperature}, Top-p: {request.top_p}, Max tokens: {request.max_tokens}")
        
        result = get_azure_response_with_hyperparameters(
            message=request.message,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens
        )
        
        logger.info(f"Response generated - Length: {result['response_length']} characters, Tokens: {result['token_count']}")
        
        return HyperparameterResponse(
            response=result["response"],
            parameters_used=result["parameters_used"],
            response_length=result["response_length"],
            token_count=result["token_count"]
        )
    
    except Exception as e:
        logger.error(f"Hyperparameter endpoint error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Azure OpenAI Hyperparameter Demo")
    uvicorn.run(app, host="0.0.0.0", port=8000)
