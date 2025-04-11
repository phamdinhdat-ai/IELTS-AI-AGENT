import os
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import AnyHttpUrl # Use for URL validation

class Settings(BaseSettings):
    PROJECT_NAME: str = "KnowledgeSphere Agentic RAG"
    API_V1_STR: str = "/api/v1"

    # --- Database ---
    DATABASE_URL: str = "postgresql+psycopg://datpd1:datpd1@localhost:5432/knowledge_db"

    # --- Security ---
    SECRET_KEY: str = "datpd1"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    ALGORITHM: str = "HS256"

    # --- Models & RAG ---
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/bge-base-en-v1.5"
    EMBEDDING_DIMENSION: int = 768 # Must match the model!
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    RAG_RETRIEVER_K: int = 3

    # --- LLM ---
    OLLAMA_BASE_URL: AnyHttpUrl="http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.1"
    LLM_TEMPERATURE: float = 0.1

    # --- CORS ---
    # Accept comma-separated string from env and convert to list
    BACKEND_CORS_ORIGINS_STR: str = "http://localhost:8000"

    @property
    def BACKEND_CORS_ORIGINS(self) -> list[AnyHttpUrl]:
        return [origin.strip() for origin in self.BACKEND_CORS_ORIGINS_STR.split(",") if origin]

    # Load .env file relative to this config file's directory is usually robust
    # Or rely on dotenv being loaded before this is imported
    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(__file__), "..", "..", ".env"), # Path to .env
        env_file_encoding='utf-8',
        case_sensitive=True,
        extra='ignore'
    )

settings = Settings()

# Optional: Add logging here if needed early
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Settings loaded successfully.")