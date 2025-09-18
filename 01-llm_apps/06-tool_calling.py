import os
import json
import logging
import httpx
import ipaddress
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from openai import AzureOpenAI
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
# Handle both direct execution and uvicorn execution
current_dir = Path(__file__).parent
project_root = current_dir.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=str(env_path))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Azure OpenAI Tool Calling Demo", version="1.0.0")

class ChatRequest(BaseModel):
    message: str
    user_ip: Optional[str] = None

class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any]
    result: Any

class ChatResponse(BaseModel):
    response: str
    tools_used: List[ToolCall] = []
    user_location: Optional[Dict[str, Any]] = None

def initialize_azure_client():
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    
    # Load additional API keys (support common variants)
    apiip_key = os.getenv("APIIP_KEY") or os.getenv("APIP_KEY") or os.getenv("APIP_API_KEY")
    
    if not all([azure_endpoint, azure_api_key, azure_api_version, azure_deployment]):
        raise ValueError("Missing required Azure OpenAI environment variables. Please check your .env file.")
    
    if not apiip_key:
        logger.warning("Neither APIIP_KEY, APIP_KEY nor APIP_API_KEY found in environment variables. IP geolocation will not work.")
    
    # Weather functionality removed; no AccuWeather key needed
    
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

# Tool Functions

async def get_location_by_ip(ip_address: str) -> Dict[str, Any]:
    """
    Get location information based on IP address using APIIP service.
    """
    try:
        apiip_key = os.getenv("APIIP_KEY") or os.getenv("APIP_KEY") or os.getenv("APIP_API_KEY")
        if not apiip_key:
            raise ValueError("APIIP_KEY not found")
        
        async with httpx.AsyncClient() as client:
            url = "https://apiip.net/api/check"
            params = {
                "ip": ip_address,
                "accessKey": apiip_key
            }
            
            logger.info(f"Getting location for IP: {ip_address}")
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            return {
                "ip": data.get("ip"),
                "city": data.get("city"),
                "region": data.get("regionName"),
                "country": data.get("countryName"),
                "country_code": data.get("countryCode"),
                "latitude": data.get("latitude"),
                "longitude": data.get("longitude"),
                "timezone": data.get("timezoneName"),
                "isp": data.get("ispName")
            }
            
    except Exception as e:
        logger.error(f"IP Location API Error: {str(e)}")
        return {"error": f"Failed to get location data: {str(e)}"}

IP_LOCATION_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_location_by_ip",
        "description": "Get geographic location information based on an IP address for personalization",
        "parameters": {
            "type": "object",
            "properties": {
                "ip_address": {
                    "type": "string",
                    "description": "The IP address to get location information for"
                }
            },
            "required": ["ip_address"]
        }
    }
}

AVAILABLE_TOOLS = [IP_LOCATION_TOOL_SCHEMA]

def get_client_ip(request: Request) -> str:
    """Extract client IP address from request headers."""
    # Check for forwarded IP first (common in production with load balancers)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Take the first IP if multiple are present
        return forwarded_for.split(",")[0].strip()
    
    # Check for real IP header
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    # Fall back to direct client IP
    return request.client.host if request.client else "127.0.0.1"

def is_private_or_local(ip: str) -> bool:
    """Return True if the IP is localhost or from a private range."""
    if not ip:
        return True
    if ip == "::1" or ip.startswith("127."):
        return True
    parts = ip.split(".")
    if len(parts) == 4:
        try:
            first = int(parts[0])
            second = int(parts[1])
            if first == 10:
                return True
            if first == 192 and second == 168:
                return True
            if first == 172 and 16 <= second <= 31:
                return True
        except ValueError:
            return False
    return False

async def get_public_ip() -> str:
    """Fetch the public IP of this machine for local testing."""
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get("https://api.ipify.org", params={"format": "json"}, timeout=5.0)
            r.raise_for_status()
            return r.json().get("ip", "")
    except Exception:
        return ""

async def execute_tool_call(tool_name: str, arguments: Dict[str, Any]) -> Any:
    """Execute a tool call and return the result."""
    if tool_name == "get_location_by_ip":
        return await get_location_by_ip(arguments["ip_address"])
    else:
        return {"error": f"Unknown tool: {tool_name}"}

 

@app.post("/chat", response_model=ChatResponse)
async def chat_with_tools(request: ChatRequest, http_request: Request):
    """
    Main chat endpoint that supports tool calling for weather and location services.
    """
    try:
        logger.info(f"Chat request received - Message: {request.message[:50]}...")
        
        # Extract client IP if not provided and compute an effective public IP for local testing
        client_ip = request.user_ip or get_client_ip(http_request)
        effective_ip = client_ip
        if is_private_or_local(effective_ip):
            fallback_ip = await get_public_ip()
            if fallback_ip:
                effective_ip = fallback_ip
        logger.info(f"Client IP (reported): {client_ip} | (effective): {effective_ip}")
        
        # Prepare messages for OpenAI
        messages = [
            {
                "role": "system",
                "content": """You are a helpful assistant with access to a location tool.

TOOL USAGE POLICY (IMPORTANT):
- If a user request is location-dependent or mentions phrases like: 'my place', 'my city', 'my location', 'near me', 'nearby', 'here', 'around me', 'local', 'in my city', 'in my area', you MUST first determine the user's location by calling the get_location_by_ip tool.
- If the user's city/town/region is explicitly provided in the message, DO NOT call the tool for IP location; use the provided location instead.
- If you do not have the user's IP in the conversation, call get_location_by_ip with ip_address set to 'user_ip' and the server will substitute the real client IP.
- After obtaining location data, incorporate the city/region/country naturally in your answer. Do NOT reveal the raw IP address.
- If the tool fails (or returns an error), then politely ask the user for their city or location and continue.

Be concise, accurate, and helpful in your responses."""
            },
            {"role": "user", "content": request.message}
        ]
        
        deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        if not deployment_name:
            raise ValueError("AZURE_OPENAI_DEPLOYMENT_NAME environment variable is not set")
        
        # Make initial OpenAI call with tools; model will decide based on system prompt
        response = azure_client.chat.completions.create(
            model=deployment_name,
            messages=messages,
            tools=AVAILABLE_TOOLS,
            tool_choice="auto"
        )
        
        response_message = response.choices[0].message
        tools_used = []
        user_location = None
        
        # Handle tool calls if present
        if response_message.tool_calls:
            logger.info(f"Tool calls detected: {len(response_message.tool_calls)}")
            
            # Add assistant message to conversation
            messages.append(response_message)
            
            # Execute each tool call
            for tool_call in response_message.tool_calls:
                tool_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                
                # Inject or correct IP when the model triggers location lookup
                if tool_name == "get_location_by_ip":
                    ip_arg = (arguments.get("ip_address") or "").strip()
                    placeholders = {
                        "user_ip", "<user_ip>", "{user_ip}",
                        "client_ip", "<client_ip>", "{client_ip}",
                        "request_ip", "ip", "localhost"
                    }
                    replace = False
                    if not ip_arg or ip_arg.lower() in placeholders:
                        replace = True
                    else:
                        try:
                            _ = ipaddress.ip_address(ip_arg)
                            # Replace if private/loopback
                            if is_private_or_local(ip_arg):
                                replace = True
                        except ValueError:
                            replace = True
                    if replace:
                        arguments["ip_address"] = effective_ip

                logger.info(f"Executing tool: {tool_name} with args: {arguments}")
                
                # Execute the tool
                tool_result = await execute_tool_call(tool_name, arguments)
                
                # Store tool usage for response
                tools_used.append(ToolCall(
                    name=tool_name,
                    arguments=arguments,
                    result=tool_result
                ))
                
                # If this was an IP location call, store the location
                if tool_name == "get_location_by_ip" and "error" not in tool_result:
                    user_location = tool_result
                
                # Add tool result to conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_result)
                })
            
            # Get final response from OpenAI with tool results
            final_response = azure_client.chat.completions.create(
                model=deployment_name,
                messages=messages
            )
            
            final_message = final_response.choices[0].message.content
        else:
            final_message = response_message.content
        
        logger.info(f"Response generated with {len(tools_used)} tools used")
        
        return ChatResponse(
            response=final_message,
            tools_used=tools_used,
            user_location=user_location
        )
    
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Azure OpenAI Tool Calling Demo")
    uvicorn.run(app, host="0.0.0.0", port=8000)

