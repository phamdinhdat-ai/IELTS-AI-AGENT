import logging
from typing import List, Dict, Any, Tuple

from sqlalchemy.ext.asyncio import AsyncSession

from core.config import settings
from db.models.user import User, UserRole
from db.models.knowledge import KnowledgeChunk
from crud import crud_knowledge, crud_chat # crud_chat will be created next
from schemas.chat import ChatMessageCreate, SourceDocument # Import necessary schemas

# Langchain and Model Loading (Load once, reuse)
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

# --- Initialize Models (Load once globally or use FastAPI lifespan events) ---
# It's generally better to load models during app startup (in main.py or lifespan)
# For simplicity here, we load them when the service module is imported.
# Ensure thread safety if needed in a multi-process environment.

try:
    logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL_NAME}")
    # Ensure CUDA is available if running on GPU, default is CPU
    encode_kwargs = {'normalize_embeddings': True} # Normalize for cosine similarity
    embedding_model = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu', 'trust_remote_code':True}, # Change to 'cuda' if needed
        encode_kwargs=encode_kwargs,
    )
    logger.info("Embedding model loaded successfully.")

    logger.info(f"Initializing LLM: {settings.OLLAMA_MODEL} at {settings.OLLAMA_BASE_URL}")
    llm = ChatOllama(
        model= str(settings.OLLAMA_MODEL),
        # base_url= str(settings.OLLAMA_BASE_URL), # Pydantic URL needs casting back to str
        temperature=settings.LLM_TEMPERATURE
    )
    # Perform a quick test invocation during initialization
    llm.invoke("Initialize test") # Can be slow, consider optional/async startup check
    logger.info("LLM initialized (connection test skipped in service init).")

except Exception as e:
    logger.exception(f"Failed to initialize AI models: {e}")
    # Depending on policy, you might want to raise the exception
    # or allow the app to start but log critical errors.
    embedding_model = None
    llm = None

# --- Prompt Template ---
# Adjust this prompt based on experimentation with Mistral Instruct
RAG_PROMPT_TEMPLATE = """
CONTEXT:
{context}

CONVERSATION HISTORY:
{chat_history}

QUESTION:
{question}

INSTRUCTIONS:
- You are a helpful assistant for KnowledgeSphere.
- Answer the QUESTION based ONLY on the provided CONTEXT above.
- If the CONTEXT doesn't contain the answer, state clearly that you cannot answer based on the provided documents. Do not make up information.
- If CONTEXT is available, synthesize the information relevant to the QUESTION.
- Keep the answer concise and informative (max 3-4 sentences unless details are specifically requested).
- Mention the source document(s) briefly if appropriate (e.g., "According to policy_hr.txt..."). Do NOT directly quote the CONTEXT verbatim unless necessary.
- Ignore the CONVERSATION HISTORY if it seems irrelevant to the current QUESTION.

ANSWER:
"""

rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
output_parser = StrOutputParser()

# Combine prompt, LLM, and parser into a basic chain
# Note: Context retrieval happens *before* this chain is invoked
rag_chain = rag_prompt | llm | output_parser

# --- Helper Functions ---

def _format_chat_history(history: List[Tuple[str, str]]) -> str:
    """Formats chat history list into a string for the prompt."""
    if not history:
        return "No previous conversation history."
    buffer = ""
    for role, content in history:
        buffer += f"{role.capitalize()}: {content}\n"
    return buffer

def _format_context(docs: List[KnowledgeChunk]) -> str:
    """Formats retrieved document chunks into a string for the prompt."""
    if not docs:
        return "No relevant documents found."
    context_str = ""
    for i, doc in enumerate(docs):
        source_name = doc.metadata_.get('source', f'Document Chunk {doc.id}') if doc.metadata_ else f'Document Chunk {doc.id}'
        content_snippet = doc.content # Include full content for the LLM
        context_str += f"--- Source {i+1}: {source_name} ---\n{content_snippet}\n\n"
    return context_str

# --- Main Service Function ---

async def get_rag_response(
    db: AsyncSession,
    query: str,
    user: User,
    session_id: str # Used to scope conversation history
) -> Dict[str, Any]:
    """
    Gets a RAG response, handling retrieval, generation, and history.
    """
    if not embedding_model or not llm:
        logger.error("RAG service cannot function: AI models not loaded.")
        raise ConnectionError("AI models are not available.") # Or return specific error response

    logger.info(f"Processing RAG query for user {user.id} (role: {user.role}), session: {session_id}")

    # 1. Generate Query Embedding
    try:
        query_embedding = embedding_model.embed_query(query)
        logger.debug("Generated query embedding.")
    except Exception as e:
        logger.exception(f"Failed to generate query embedding: {e}")
        raise

    # 2. Retrieve Relevant Chunks (with authorization)
    try:
        relevant_chunks = await crud_knowledge.get_relevant_chunks_for_user(
            db=db,
            embedding=query_embedding,
            user_role=user.role,
            limit=settings.RAG_RETRIEVER_K
            # Add min_similarity if needed: min_similarity=0.7
        )
        logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks.")
    except Exception as e:
        logger.exception(f"Failed during knowledge retrieval: {e}")
        raise

    # 3. Format Context for LLM
    context_str = _format_context(relevant_chunks)

    # 4. Get Conversation History (Requires crud_chat)
    # Placeholder: Implement history fetching using session_id and user_id
    # For this example, we'll use an empty history
    chat_history_tuples: List[Tuple[str, str]] = [] # Replace with actual DB call result
    # Example DB call:
    # chat_history_models = await crud_chat.get_messages(db, session_id=session_id, user_id=user.id, limit=5)
    # chat_history_tuples = [(msg.role, msg.content) for msg in chat_history_models]
    formatted_history = _format_chat_history(chat_history_tuples)
    logger.debug(f"Formatted history for prompt:\n{formatted_history}")

    # 5. Invoke RAG Chain (LLM Call)
    try:
        logger.debug("Invoking RAG chain with LLM...")
        chain_input = {
            "context": context_str,
            "chat_history": formatted_history,
            "question": query
        }
        answer = await rag_chain.ainvoke(chain_input) # Use async invoke
        logger.info(f"LLM generated answer snippet: {answer[:100]}...")
    except Exception as e:
        logger.exception(f"Error invoking LLM chain: {e}")
        # Consider returning a specific error message to the user
        answer = "Sorry, I encountered an error trying to generate a response."
        # Or raise the exception to be handled by the API layer

    # 6. Store Interaction in DB (Requires crud_chat)
    # Placeholder: Implement storing user query and AI response
    # Example DB call:
    # await crud_chat.create_message(db, session_id=session_id, user_id=user.id, role="user", content=query)
    # await crud_chat.create_message(db, session_id=session_id, user_id=user.id, role="assistant", content=answer, sources_metadata=[c.metadata_ for c in relevant_chunks])

    # 7. Format Sources for API Response
    sources = [SourceDocument.from_orm(chunk) for chunk in relevant_chunks]

    return {
        "answer": answer,
        "sources": sources,
        "session_id": session_id # Return session ID for client tracking
    }