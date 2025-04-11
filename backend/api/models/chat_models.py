from pydantic import BaseModel, Field
from typing import List, Optional



# --- API Models ---
class ChatRequest(BaseModel):
    query: str
    session_id: str = "default_session" # Provide a session ID for history

class SourceDocument(BaseModel):
    source: Optional[str] = None
    content: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    session_id: str