from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel # For defining request/response data shapes

# --- Input/Output Models (using Pydantic) ---
class ChatRequest(BaseModel):
    """Request model for incoming chat messages."""
    message: str
    user_id: str | None = None # Optional user ID for context later

class ChatResponse(BaseModel):
    """Response model for outgoing chat messages."""
    reply: str
    # Add more fields later, like session_id, suggested_replies, etc.



# --- Input/Output Models for Speech (Example) ---
class ASRResponse(BaseModel):
    """Response model for Automatic Speech Recognition."""
    transcript: str
    confidence: float | None = None # Optional confidence score
