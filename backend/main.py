# main.py
import os
import json
import logging
from datetime import timedelta
from typing import List, Dict, Any, Optional, Annotated # Added Annotated

from dotenv import load_dotenv

# FastAPI and Pydantic
from fastapi import FastAPI, HTTPException, Request, Header, Depends # Added Depends
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm # For login form data
# from pydantic import BaseModel
from langchain_core.pydantic_v1 import BaseModel # Use langchain_core for Pydantic

from fastapi import status
# Langchain components (as before)
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.react.agent import create_react_agent
from langchain.agents import AgentExecutor

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain import hub # Added import for hub.pull

# Custom Tools, User DB, Security
from tools.custom_tools import get_customer_tools, available_tools
import tools.custom_tools as custom_tools_module
from user_db import UserInDB, get_user, fake_users_db # Import simulated DB
from user_db import UserInDBBase # Import base model for user data
from security import ( # Import security functions
    create_access_token,
    get_current_active_user,
    verify_password,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    TokenData # Import TokenData if needed, although handled by get_current_user
)

# --- Configuration Loading ---
load_dotenv()
# (Keep existing config loading: PERSIST_DIRECTORY_BASE, EMBEDDING_MODEL_NAME, etc.)
PERSIST_DIRECTORY_BASE = os.getenv("PERSIST_DIRECTORY_BASE", "chroma_dbs")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/bge-base-en-v1.5")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.getenv("DEFAULT_OLLAMA_MODEL", "mistral")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.1))
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
CUSTOMER_CONFIG_DIR = "customer_configs"

# --- Logging Setup ---
# (Keep existing logging setup)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Variables / Caches ---
# (Keep existing caches: customer_configs, vector_stores, llm_instances)
# (Keep embedding_function global initialization)
customer_configs: Dict[str, Dict] = {}
vector_stores: Dict[str, Chroma] = {}
llm_instances: Dict[str, ChatOllama] = {}
embedding_function = None
# Modify session history store: Key is now user_id
# Structure: { user_id: ChatMessageHistory } - Simpler for this example
user_chat_histories: Dict[int, ChatMessageHistory] = {}

# --- Helper Functions ---
# (Keep load_customer_config, get_llm_instance - they are still needed)
# (Keep get_vector_store - it now uses customer_id from config)
# (Modify initialize_agent_executor slightly if needed, but structure is similar)

def load_customer_config(customer_id: str) -> Optional[Dict]:
    # (Keep implementation as before)
    if customer_id in customer_configs: return customer_configs[customer_id]
    config_path = os.path.join(CUSTOMER_CONFIG_DIR, f"{customer_id}.json")
    if not os.path.exists(config_path): return None
    try:
        with open(config_path, 'r') as f: config = json.load(f)
        customer_configs[customer_id] = config
        logger.info(f"Loaded configuration for customer: {customer_id}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration for customer {customer_id}: {e}")
        return None

def get_llm_instance(config: Dict) -> ChatOllama:
    # (Keep implementation as before)
    model_name = config.get("llm_model", DEFAULT_OLLAMA_MODEL)
    instance_key = model_name
    if instance_key in llm_instances: return llm_instances[instance_key]
    logger.info(f"Initializing LLM (Model: {model_name}, Base URL: {OLLAMA_BASE_URL})")
    llm = ChatOllama(model=model_name, base_url=OLLAMA_BASE_URL, temperature=LLM_TEMPERATURE)
    try:
        llm.invoke("Test prompt: Hello!")
        logger.info(f"LLM connection successful for model {model_name}.")
        llm_instances[instance_key] = llm
        return llm
    except Exception as e:
        logger.error(f"Failed to connect to LLM model {model_name}. Error: {e}")
        raise

def get_vector_store(config: Dict) -> Optional[Chroma]:
    # (Keep implementation, uses config["customer_id"] and config["kb_collection_name"])
    global embedding_function
    customer_id = config["customer_id"]
    collection_name = config.get("kb_collection_name", f"{customer_id}_default_kb") # Default name if missing
    persist_dir = os.path.join(PERSIST_DIRECTORY_BASE, customer_id)
    if collection_name in vector_stores: return vector_stores[collection_name]
    if embedding_function is None:
       logger.info(f"Initializing embedding function: {EMBEDDING_MODEL_NAME}")
       encode_kwargs = {'normalize_embeddings': True}
       embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={'device': 'cpu'}, encode_kwargs=encode_kwargs)
    if not os.path.exists(persist_dir):
        logger.warning(f"Persistence directory {persist_dir} not found for {customer_id}. Running ingestion...")
        customer_data_path = os.path.join("poc_kbs", f"{customer_id}_data")
        if not os.path.exists(customer_data_path):
             logger.error(f"Data path {customer_data_path} not found for {customer_id}.")
             return None
        logger.info(f"Ingesting data from {customer_data_path} for {collection_name}")
        loader = DirectoryLoader(customer_data_path, glob="**/*.*", loader_cls=TextLoader, show_progress=True, use_multithreading=True, loader_kwargs={'autodetect_encoding': True}) # Allow any text-like file
        documents = loader.load()
        if not documents:
             logger.warning(f"No documents found in {customer_data_path}")
             return None
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Creating new vector store for {customer_id} in {persist_dir} collection {collection_name}")
        db = Chroma.from_documents(documents=chunks, embedding=embedding_function, collection_name=collection_name, persist_directory=persist_dir)
        logger.info(f"Vector store created for {customer_id}.")
    else:
        logger.info(f"Loading existing vector store for {customer_id} from {persist_dir} collection {collection_name}")
        db = Chroma(collection_name=collection_name, persist_directory=persist_dir, embedding_function=embedding_function)
    vector_stores[collection_name] = db
    return db

def initialize_agent_executor(config: Dict, llm: ChatOllama, vector_store: Optional[Chroma], current_user: UserInDB) -> AgentExecutor:
    """Initializes the AgentExecutor for the customer, potentially passing user context."""
    customer_id = config["customer_id"]
    allowed_tool_names = config.get("allowed_tool_names", [])
    system_prompt_template = config.get("agent_system_prompt", "You are a helpful assistant.")

    # --- Inject User/Customer Context into System Prompt ---
    # Example: Add user name and customer name to the prompt
    system_prompt = system_prompt_template + f"\nYou are assisting {current_user.full_name or current_user.username} from {config.get('customer_name', customer_id)}."

    # --- Tool Initialization with Context (Advanced - Requires Tool Refactoring) ---
    # Proper way: Initialize tool instances here, passing user_id or other context
    # Example:
    # user_specific_tools = []
    # if "user_order_tool" in allowed_tool_names:
    #     user_specific_tools.append(UserOrderTool(user_id=current_user.user_id))

    # --- Hacky Retriever Injection (Still needs fixing for production) ---
    if vector_store:
        custom_tools_module.customer_retriever_placeholder = vector_store.as_retriever(search_kwargs={"k": 3})
        logger.info(f"KB tool initialized for customer {customer_id}")
    else:
        if "customer_knowledge_base" in allowed_tool_names:
            logger.warning(f"Disabling 'customer_knowledge_base' tool for {customer_id} - vector store unavailable.")
            allowed_tool_names = [name for name in allowed_tool_names if name != "customer_knowledge_base"]
        custom_tools_module.customer_retriever_placeholder = None
    # --- End Hack ---

    tools = get_customer_tools(allowed_tool_names) # Gets tool functions/objects
    # tools.extend(user_specific_tools) # Add user-specific tool instances if created

    if not tools:
        logger.warning(f"No tools available for customer {customer_id}.")

    # Use the updated system prompt in the agent's base prompt
    react_prompt = hub.pull("hwchase17/react-chat")
    # Inject the dynamic system prompt
    # This requires modifying the pulled prompt structure or creating a custom one
    # Assuming react_prompt has a structure allowing system message update/replacement
    # Simplified approach: Add system message separately if agent supports it
    # Or, modify the template string directly if needed.
    # For ChatOllama with ReAct, system message might be part of the prompt template itself.

    agent = create_react_agent(llm, tools, react_prompt)

    # Agent Executor - Memory will be handled by RunnableWithMessageHistory wrapper
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )
    return agent_executor

# Function to get user-specific memory
def get_user_memory(user_id: int) -> ConversationBufferMemory:
     if user_id not in user_chat_histories:
         user_chat_histories[user_id] = ChatMessageHistory()
     # Return memory instance linked to the specific user's history store
     return ConversationBufferMemory(
        chat_memory=user_chat_histories[user_id],
        memory_key='chat_history', # Must match key expected by agent/prompt
        return_messages=True
     )


# --- FastAPI Application ---
app = FastAPI(title="KnowledgeSphere Agentic API with Auth")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- API Models ---
class Token(BaseModel):
    access_token: str
    token_type: str

class AgentChatRequest(BaseModel):
    query: str
    # session_id is removed, identified by user token

class AgentChatResponse(BaseModel):
    answer: str
    # session_id: str # Removed
    intermediate_steps: Optional[List[Dict]] = None

# --- Lifespan Events ---
@app.on_event("startup")
async def startup_event():
    # (Keep existing startup logic: init embedding_function, ensure dirs exist)
    global embedding_function
    logger.info("Application startup...")
    if embedding_function is None:
       logger.info(f"Initializing embedding function: {EMBEDDING_MODEL_NAME}")
       encode_kwargs = {'normalize_embeddings': True}
       embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={'device': 'cpu'}, encode_kwargs=encode_kwargs)
    os.makedirs(PERSIST_DIRECTORY_BASE, exist_ok=True)
    os.makedirs(CUSTOMER_CONFIG_DIR, exist_ok=True)
    logger.info("Startup complete.")


# --- Authentication Endpoints ---
@app.post("/token", response_model=Token, tags=["Authentication"])
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()]
):
    """Login endpoint to get JWT token."""
    user = get_user(form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=UserInDBBase, tags=["Users"])
async def read_users_me(
    current_user: Annotated[UserInDB, Depends(get_current_active_user)]
):
    """Returns details of the currently logged-in user."""
    # Return model excluding password hash
    return UserInDBBase.model_validate(current_user)


# --- Agent Chat Endpoint (Protected) ---
@app.post("/agent/chat", response_model=AgentChatResponse, tags=["Agent"])
async def agent_chat_endpoint(
    request: AgentChatRequest,
    current_user: Annotated[UserInDB, Depends(get_current_active_user)] # Require authentication
):
    customer_id = current_user.customer_id
    user_id = current_user.user_id
    logger.info(f"Agent query for user '{current_user.username}' (Cust: {customer_id}): {request.query}")

    # 1. Load Customer Config
    config = load_customer_config(customer_id)
    if not config:
        raise HTTPException(status_code=503, detail=f"Configuration unavailable for customer: {customer_id}")

    try:
        # 2. Get/Initialize LLM
        llm = get_llm_instance(config)

        # 3. Get/Initialize Vector Store
        vector_store = get_vector_store(config)

        # 4. Initialize Agent Executor
        agent_executor = initialize_agent_executor(config, llm, vector_store, current_user)

        # 5. Get User-Specific Memory
        memory = get_user_memory(user_id)

        # 6. Wrap Agent with History using RunnableWithMessageHistory
        agent_with_chat_history = RunnableWithMessageHistory(
            agent_executor,
            lambda session_id: user_chat_histories[int(session_id)], # Function to get history store by session_id (using user_id as session_id)
            input_messages_key="input",
            history_messages_key="chat_history",
        )

        # 7. Invoke Agent with history management wrapper
        # Pass user_id as the 'session_id' for the history wrapper
        response = await agent_with_chat_history.ainvoke(
            {"input": request.query},
            config=RunnableConfig(configurable={"session_id": str(user_id)})
        )

        logger.info(f"Agent response for user '{current_user.username}': {response.get('output', '')[:100]}...")

        intermediate_steps = response.get("intermediate_steps", [])
        formatted_steps = [{"tool": step[0].tool, "input": step[0].tool_input, "output": step[1]} for step in intermediate_steps]

        return AgentChatResponse(
            answer=response.get("output", "Agent did not produce a final answer."),
            intermediate_steps=formatted_steps
        )

    except Exception as e:
        logger.exception(f"Error processing agent request for user '{current_user.username}': {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


# --- Static Files (Serve Frontend) ---
app.mount("/", StaticFiles(directory="static", html=True), name="static")

# --- Main Execution Guard ---
if __name__ == "__main__":
    import uvicorn
    os.makedirs(PERSIST_DIRECTORY_BASE, exist_ok=True)
    os.makedirs(CUSTOMER_CONFIG_DIR, exist_ok=True)
    logger.info(f"Starting Uvicorn server on {API_HOST}:{API_PORT}")
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=True)