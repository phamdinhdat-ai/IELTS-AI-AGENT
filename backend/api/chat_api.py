import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Langchain components
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader # Add loaders as needed
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama # Langchain integration for Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory # In-memory history

# --- Configuration Loading ---
load_dotenv()

DATA_PATH = os.getenv("DATA_PATH", "data")
PERSIST_DIRECTORY = os.getenv("PERSIST_DIRECTORY", "chroma_db_persist")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/bge-base-en-v1.5")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.1))
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Variables ---
vector_db = None
chat_model = None
conversational_rag_chain = None
# Simple in-memory store for session histories
session_histories: Dict[str, ChatMessageHistory] = {}

# --- Helper Functions ---

def load_documents():
    """Loads documents from various formats."""
    # Simple loader for PoC, extend with PyPDFLoader, etc. as needed
    text_loader_kwargs={'autodetect_encoding': True}
    loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.txt", # Load .txt files
        loader_cls=TextLoader,
        loader_kwargs=text_loader_kwargs,
        show_progress=True,
        use_multithreading=True
    )
    # Add other loaders here, e.g., for PDF:
    pdf_loader = DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True)
    loaded_pdfs = pdf_loader.load()

    logger.info(f"Loading documents from {DATA_PATH}")
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} text documents.")
    documents.extend(loaded_pdfs) # Combine if using multiple loaders
    return documents

def split_documents(documents):
    """Splits documents into chunks."""
    logger.info(f"Splitting {len(documents)} documents into chunks (size: {CHUNK_SIZE}, overlap: {CHUNK_OVERLAP})")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks.")
    return chunks

def initialize_vector_db(chunks):
    """Initializes ChromaDB with document chunks."""
    logger.info(f"Initializing vector store with model: {EMBEDDING_MODEL_NAME}")
    # Ensure CUDA is available if running on GPU, default is CPU
    encode_kwargs = {'normalize_embeddings': True} # Often recommended for cosine similarity
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}, # Change to 'cuda' if GPU available and configured
        encode_kwargs=encode_kwargs
    )

    # Create or load the vector store
    if os.path.exists(PERSIST_DIRECTORY):
        logger.info(f"Loading existing vector store from: {PERSIST_DIRECTORY}")
        db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
    else:
        logger.info(f"Creating new vector store in: {PERSIST_DIRECTORY}")
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIRECTORY
        )
        logger.info("Vector store created and persisted.")
    return db

def initialize_llm():
    """Initializes the Ollama LLM."""
    logger.info(f"Initializing LLM (Model: {OLLAMA_MODEL}, Base URL: {OLLAMA_BASE_URL})")
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=LLM_TEMPERATURE
    )
    # Simple test call
    try:
         llm.invoke("Test prompt: Hello!")
         logger.info("LLM connection successful.")
    except Exception as e:
         logger.error(f"Failed to connect to LLM at {OLLAMA_BASE_URL}. Ensure Ollama is running. Error: {e}")
         raise
    return llm

def setup_rag_chain(llm, retriever):
    """Sets up the conversational RAG chain."""
    logger.info("Setting up RAG chain...")

    # Prompt Template for RAG
    # Make sure the variable names match: {chat_history}, {context}, {input}
    template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    Context: {context}

    Chat History:
    {chat_history}

    Question: {input}
    Answer:"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        #MessagesPlaceholder(variable_name="chat_history"), # Handled manually for more control below
        ("human", "{input}")
    ])

    output_parser = StrOutputParser()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # RAG chain structure
    rag_chain_from_docs = (
        {
            "context": lambda input_dict: format_docs(input_dict["documents"]),
            "input": lambda input_dict: input_dict["input"],
            "chat_history": lambda input_dict: format_history(input_dict["chat_history"])
        }
        | prompt
        | llm
        | output_parser
    )

    # Function to handle session history retrieval
    def get_session_history(session_id: str) -> ChatMessageHistory:
        if session_id not in session_histories:
            session_histories[session_id] = ChatMessageHistory()
        return session_histories[session_id]

    def format_history(messages: List[Any]) -> str:
        """ Formats history messages into a string for the prompt. """
        return "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in messages])


    # Wrap the chain with history management
    # Note: We manually fetch history and format context within the chain itself now
    # Langchain's RunnableWithMessageHistory can be complex, doing it manually here
    # for clarity in passing context AND history AND input.

    # We will handle history outside the core chain call in the endpoint
    logger.info("RAG chain setup complete.")
    return rag_chain_from_docs # Return the core chain part

# --- FastAPI Application ---

@asynccontextmanager
async def startup_event(app: FastAPI):
    global vector_db, chat_model, conversational_rag_chain
    logger.info("Application startup: Initializing models and vector DB...")
    try:
        # 1. Load and process documents only if DB doesn't exist
        if not os.path.exists(PERSIST_DIRECTORY):
            documents = load_documents()
            chunks = split_documents(documents)
            vector_db = initialize_vector_db(chunks)
        else:
            # Just load the existing DB
            logger.info(f"Loading existing vector store from: {PERSIST_DIRECTORY}")
            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={'device': 'cpu'})
            vector_db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
            logger.info("Vector DB loaded.")

        # 2. Initialize LLM
        chat_model = initialize_llm()

        # 3. Setup RAG chain
        retriever = vector_db.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 chunks
        conversational_rag_chain = setup_rag_chain(chat_model, retriever) # Pass retriever if needed inside setup

        logger.info("Startup complete. API is ready.")

    except Exception as e:
        logger.exception(f"Critical error during startup: {e}")
        # Depending on the error, you might want the app to fail startup
        raise HTTPException(status_code=500, detail=f"Application failed to initialize: {e}")



app = FastAPI(title="KnowledgeSphere RAG API", lifespan=startup_event)

# CORS Middleware (allow requests from frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for development, restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


# --- Lifespan Events (Load models on startup) ---

# --- API Endpoints ---
@app.get("/health", summary="Check API Health")
async def health_check():
    # Add more checks if needed (DB connection, LLM ping)
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse, summary="Handle Chat Queries")
async def chat_endpoint(request: ChatRequest):
    global conversational_rag_chain, vector_db
    logger.info(f"Received query for session '{request.session_id}': {request.query}")

    if vector_db is None or chat_model is None or conversational_rag_chain is None:
        raise HTTPException(status_code=503, detail="System not initialized yet.")

    try:
        # 1. Retrieve session history
        session_id = request.session_id
        if session_id not in session_histories:
            session_histories[session_id] = ChatMessageHistory()
        current_history = session_histories[session_id]
        formatted_history_for_prompt = format_history(current_history.messages)

        # 2. Retrieve relevant documents
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        retrieved_docs = retriever.invoke(request.query)
        logger.info(f"Retrieved {len(retrieved_docs)} documents for query.")

        # 3. Invoke the RAG chain
        inputs = {
            "input": request.query,
            "documents": retrieved_docs,
            "chat_history": formatted_history_for_prompt # Pass history for the prompt context
        }
        answer = conversational_rag_chain.invoke(inputs) # This chain uses the context/history within its structure now
        logger.info(f"Generated answer for session '{session_id}': {answer[:100]}...") # Log snippet

        # 4. Update history AFTER getting the response
        current_history.add_user_message(request.query)
        current_history.add_ai_message(answer)

        # 5. Prepare source documents for response
        sources = [
            SourceDocument(
                source=doc.metadata.get('source', 'Unknown'),
                content=doc.page_content
            ) for doc in retrieved_docs
        ]

        return ChatResponse(answer=answer, sources=sources, session_id=session_id)

    except Exception as e:
        logger.exception(f"Error processing chat request for session '{request.session_id}': {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


# --- Static Files (Serve Frontend) ---
# Mount the static directory AFTER API routes to avoid conflicts
app.mount("/", StaticFiles(directory="static", html=True), name="static")


# --- Main Execution Guard ---
if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Uvicorn server on {API_HOST}:{API_PORT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT)