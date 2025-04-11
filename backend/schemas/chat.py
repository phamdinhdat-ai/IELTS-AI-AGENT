from pydantic import BaseModel, ConfigDict
from typing import List, Optional, Dict, Any
from datetime import datetime

from .knowledge import KnowledgeChunk # Keep for source representation
from db.models.chat import MessageRole # Import the Enum

# --- Schemas for Conversation Message ---
class ChatMessageBase(BaseModel):
    role: MessageRole
    content: str
    metadata_: Optional[Dict[str, Any]] = None

class ChatMessageCreate(ChatMessageBase):
    # No additional fields needed for basic creation from code
    pass

class ChatMessage(ChatMessageBase):
    id: int
    conversation_id: int
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)

# --- Schemas for Conversation ---
class ConversationBase(BaseModel):
    session_id: str

class ConversationCreate(ConversationBase):
    user_id: int # Needed when creating

class Conversation(ConversationBase):
    id: int
    user_id: int
    created_at: datetime
    updated_at: datetime
    messages: List[ChatMessage] = [] # Include messages when returning conversation details
    model_config = ConfigDict(from_attributes=True)


# --- Schemas for API Request/Response (Existing from before) ---
class ChatRequest(BaseModel):
    query: str
    # session_id is now primarily handled server-side based on user/context,
    # but client can provide one if managing sessions client-side
    session_id: Optional[str] = None

class SourceDocument(BaseModel):
    """Represents a source document chunk used for the answer."""
    document_id: Optional[str] = None
    chunk_number: Optional[int] = None
    content: str
    score: Optional[float] = None
    metadata: Optional[dict] = None

    @classmethod
    def from_knowledge_chunk(cls, chunk: KnowledgeChunk): # Renamed method
        """Helper to map from the KnowledgeChunk schema/model."""
        return cls(
            document_id=chunk.document_id,
            chunk_number=chunk.chunk_number,
            content=chunk.content,
            score=getattr(chunk, 'score', None),
            metadata=chunk.metadata_
        )

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    session_id: str # Return the session ID used/created
    # Optionally return conversation history if needed by frontend
    # history: List[ChatMessage] = []