# from langchain_ollama import ChatOllama
from langchain_community.chat_models import ChatOllama
# from core.config import Settings
from pydantic import AnyHttpUrl # Use for URL validation


class Settings:
    PROJECT_NAME: str = "KnowledgeSphere Agentic RAG"
    API_V1_STR: str = "/api/v1"

    # --- Database ---
    DATABASE_URL: str = "postgresql+psycopg://datpd1:datpd1@localhost:5432/knowledge_db"

    # --- Security ---
    SECRET_KEY: str = "datpd1"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    ALGORITHM: str = "HS256"

    # --- Models & RAG ---
    EMBEDDING_MODEL_NAME: str = "Alibaba-NLP/gte-multilingual-base"
    EMBEDDING_DIMENSION: int = 768 # Must match the model!
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    RAG_RETRIEVER_K: int = 3

    # --- LLM ---
    OLLAMA_BASE_URL: AnyHttpUrl="http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2:1b"
    LLM_TEMPERATURE: float = 0.1

    # --- CORS ---
    # Accept comma-separated string from env and convert to list
    BACKEND_CORS_ORIGINS_STR: str = "http://localhost:8000"

config = Settings()


def test_ollama_model_loader():
    

    llm = ChatOllama(
        model = str(config.OLLAMA_MODEL),
        base_url=str(config.OLLAMA_BASE_URL), 
        temperature= config.LLM_TEMPERATURE # Pydantic URL needs casting back to str
    )
    # Perform a quick test invocation during initialization
    llm.invoke("Initialize test") # Can be slow, consider optional/async startup check
    assert llm is not None
    assert llm.model == config.OLLAMA_MODEL
    assert llm.base_url == str(config.OLLAMA_BASE_URL)
    assert llm.temperature == config.LLM_TEMPERATURE
    # Test with a simple prompt
    prompt = "What is the capital of France?"
    response = llm.invoke(prompt)
    print(response)
    
if __name__ == "__main__":
    test_ollama_model_loader()
    print("Test passed!")