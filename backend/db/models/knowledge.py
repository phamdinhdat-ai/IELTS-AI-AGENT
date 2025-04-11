from datetime import datetime
import enum

from sqlalchemy import Column, Integer, String, DateTime, Text, Index
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector # Import the Vector type
from sqlalchemy.orm import Mapped # For type hinting with Vector

from db.base import Base
from core.config import settings # To get EMBEDDING_DIMENSION

# Define access levels using Python Enum
class AccessLevel(str, enum.Enum):
    public = "public"     # Accessible by customers and employees
    internal = "internal" # Accessible only by employees and admins

class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(String, index=True, nullable=False) # Identifier for the source doc
    chunk_number = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)

    # Define the embedding column using the Vector type
    # Specify the dimension from settings
    embedding: Mapped[Vector] = Column(Vector(settings.EMBEDDING_DIMENSION), nullable=True) # Made nullable temporarily

    # Store metadata like source filename, page, and access level
    metadata_ = Column("metadata", JSONB, nullable=False) # Use metadata_ to avoid Pydantic clash

    created_at = Column(DateTime, default=datetime.utcnow)

    # Add indexes for faster querying
    __table_args__ = (
        Index(
            "ix_knowledge_chunks_embedding",
            embedding,
            postgresql_using="ivfflat", # Example index type (or HNSW)
            # Adjust opclass and lists based on expected data size and query patterns
            postgresql_with={"lists": 100}
            # Note: Creating vector indexes often requires separate SQL commands after table creation
            # Or use alembic op.execute()
        ),
        Index("ix_knowledge_chunks_metadata_access_level", metadata_["access_level"]), # Index on access_level within JSONB
    )

    def __repr__(self):
        content_snippet = (self.content[:50] + '...') if len(self.content) > 50 else self.content
        return f"<KnowledgeChunk(id={self.id}, doc='{self.document_id}', chunk={self.chunk_number}, content='{content_snippet}')>"