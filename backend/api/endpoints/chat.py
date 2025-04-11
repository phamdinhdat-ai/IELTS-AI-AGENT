import logging
import uuid # For generating session IDs if needed

from fastapi import APIRouter, Depends, HTTPException, status, Body, Query
from typing import List
from api import deps # Import dependencies
from schemas import chat as chat_schema # Use alias
from services import rag_service # Import the RAG service
from crud import crud_chat # Import chat CRUD

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post(
    "/chat",
    response_model=chat_schema.ChatResponse,
    summary="Handle authenticated chat queries",
    status_code=status.HTTP_200_OK
)
async def handle_chat_query(
    *, # Force keyword arguments
    session: deps.SessionDep,
    current_user: deps.CurrentUser, # Ensures user is authenticated and active
    chat_request: chat_schema.ChatRequest = Body(...) # Get query from body
) -> chat_schema.ChatResponse:
    """
    Receives a user query, retrieves context based on user role,
    generates a response using RAG, and stores the conversation.
    """
    query = chat_request.query
    # Determine session ID: use provided or generate a new one per user conversation start
    # This logic might need refinement based on how frontend manages sessions
    session_id = chat_request.session_id
    if not session_id:
        # If no session ID provided, try to find an existing one or create new
        # This simplistic approach creates a new session per user if none provided
        # A better approach might involve storing the active session ID per user
        session_id = str(uuid.uuid4()) # Generate a new one if needed
        logger.warning(f"No session_id provided by client for user {current_user.id}, generated new: {session_id}")
        # Alternatively, raise an error if client MUST provide session_id

    logger.info(f"Chat request received: UserID={current_user.id}, SessionID={session_id}, Query='{query[:50]}...'")

    try:
        # 1. Get or create the conversation state in DB (links session_id to user)
        conversation = await crud_chat.get_or_create_conversation(
            db=session, session_id=session_id, user_id=current_user.id
        )

        # 2. Call the RAG service (which handles history fetching, retrieval, generation, saving messages)
        rag_result = await rag_service.get_rag_response(
            db=session,
            query=query,
            user=current_user, # Pass the authenticated user object (contains role)
            conversation_id=conversation.id # Pass DB conversation ID to RAG service for storing messages
        )

        # 3. Format response
        # rag_service now returns dict with 'answer', 'sources', 'session_id'
        # Map sources from service format (dict?) to API schema format if needed
        # Assuming rag_service now returns the needed structure directly based on previous step
        api_response = chat_schema.ChatResponse(
            answer=rag_result["answer"],
            sources=[chat_schema.SourceDocument.from_knowledge_chunk(s) for s in rag_result["sources"]], # Map from KnowledgeChunk
            session_id=session_id # Return the session ID used
        )
        return api_response

    except ConnectionError as e:
         logger.error(f"Connection error during chat processing: {e}")
         raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        logger.exception(f"Unhandled error during chat processing for session {session_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An internal error occurred.")

# Optional: Endpoint to get conversation history
@router.get(
    "/history/{session_id}",
    response_model=List[chat_schema.ChatMessage], # Return list of messages
    summary="Get chat history for a session"
)
async def get_chat_history(
    session_id: str,
    session: deps.SessionDep,
    current_user: deps.CurrentUser,
    limit: int = Query(50, description="Max number of messages to return"), # Add query param
):
    # Verify conversation belongs to the current user
    conversation = await crud_chat.get_or_create_conversation(db=session, session_id=session_id, user_id=current_user.id)
    if not conversation or conversation.user_id != current_user.id:
         raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found or access denied.")

    messages = await crud_chat.get_messages_for_conversation(db=session, conversation_id=conversation.id, limit=limit)
    # Pydantic will automatically serialize the list of ORM models based on ChatMessage schema
    return messages