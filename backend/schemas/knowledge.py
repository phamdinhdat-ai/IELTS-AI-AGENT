from pydantic import BaseModel, ConfigDict
from typing import Optional, Dict, Any, List
from datetime import datetime

from db.models.knowledge import AccessLevel # Import the Enum

# --- Base Schema ---
class KnowledgeChunkBase(BaseModel):
    document_id: str
    chunk_number: int
    content: str
    metadata_: Optional[Dict[str, Any]] = None # Match DB field name

# --- Schema for Creation (used by ingestion script) ---
class KnowledgeChunkCreate(KnowledgeChunkBase):
    embedding: List[float] # Expect embedding as list during creation
    metadata_: Dict[str, Any] # Make metadata mandatory on creation

# --- Schema for reading from DB ---
class KnowledgeChunkInDB(KnowledgeChunkBase):
    id: int
    created_at: datetime
    # Embedding might not always be returned in API responses unless needed
    model_config = ConfigDict(from_attributes=True)

# --- Schema for API Responses (subset of DB data) ---
class KnowledgeChunk(KnowledgeChunkBase):
    id: int
    # Optionally include a score if coming from similarity search
    score: Optional[float] = None
    model_config = ConfigDict(from_attributes=True)