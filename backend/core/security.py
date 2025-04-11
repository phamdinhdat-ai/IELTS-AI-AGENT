from pydantic import BaseModel
from typing import List, Optional

from schemas.knowledge import KnowledgeChunk # Import the schema for sources

# --- Request Schema ---
class ChatRequest(BaseModel):
    query: str
    session_id: str = "default_session" # Optional: Track conversation state

# --- Response Schema ---
class SourceDocument(BaseModel):
    """Represents a source document chunk used for the answer."""
    # Use the KnowledgeChunk schema directly or define specific fields
    # id: int
    document_id: Optional[str] = None
    chunk_number: Optional[int] = None
    content: str
    score: Optional[float] = None # Retrieval score
    metadata: Optional[dict] = None # Include relevant metadata

    @classmethod
    def from_orm(cls, chunk: KnowledgeChunk):
        """Helper to map from the ORM model."""
        return cls(
            # id=chunk.id,
            document_id=chunk.document_id,
            chunk_number=chunk.chunk_number,
            content=chunk.content,
            score=getattr(chunk, 'score', None), # Handle if score isn't always present
            metadata=chunk.metadata_ # Pass the metadata dict
        )


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    session_id: str