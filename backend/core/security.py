from datetime import datetime, timedelta, timezone
from typing import Optional, Any

from jose import jwt, JWTError
from passlib.context import CryptContext

from core.config import settings

# Setup password hashing context using bcrypt
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

ALGORITHM = settings.ALGORITHM
SECRET_KEY = settings.SECRET_KEY
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain password against a hashed password."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hashes a plain password."""
    return pwd_context.hash(password)

def create_access_token(subject: Any, expires_delta: Optional[timedelta] = None) -> str:
    """Creates a JWT access token."""
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode = {"exp": expire, "sub": str(subject)} # 'sub' claim identifies the principal (user)
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_access_token(token: str) -> Optional[str]:
    """Verifies the access token and returns the subject (user identifier) if valid."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        subject: Optional[str] = payload.get("sub")
        if subject is None:
            return None # Token is valid but missing subject
        # Optional: Add expiration check if needed, though jwt.decode handles it
        # token_data = TokenPayload(**payload) # Validate payload against schema if defined
        return subject
    except JWTError:
        # Token is invalid (expired, wrong signature, etc.)
        return None
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