import logging
from typing import List, Optional, Dict, Any, Tuple

from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload # For eager loading relationships

# Import models and schemas
from db.models.chat import Conversation, ConversationMessage, MessageRole
from schemas.chat import ChatMessageCreate # Using this for creation structure

logger = logging.getLogger(__name__)

async def get_or_create_conversation(db: AsyncSession, session_id: str, user_id: int) -> Conversation:
    """
    Gets an existing conversation by session_id and user_id,
    or creates a new one if it doesn't exist.
    """
    logger.debug(f"Getting or creating conversation for session: {session_id}, user: {user_id}")
    stmt = select(Conversation).where(
        Conversation.session_id == session_id,
        Conversation.user_id == user_id
    )
    result = await db.execute(stmt)
    conversation = result.scalars().first()

    if not conversation:
        logger.info(f"Creating new conversation for session: {session_id}, user: {user_id}")
        conversation = Conversation(session_id=session_id, user_id=user_id)
        db.add(conversation)
        await db.flush()
        await db.refresh(conversation)
    else:
        logger.debug(f"Found existing conversation ID: {conversation.id}")
    return conversation


async def create_message(
    db: AsyncSession,
    conversation_id: int,
    role: MessageRole, # Use Enum type hint
    content: str,
    metadata: Optional[Dict[str, Any]] = None # Use metadata instead of metadata_
) -> ConversationMessage:
    """Creates a new chat message associated with a conversation."""
    logger.debug(f"Creating message for conversation {conversation_id}, role: {role.value}")

    db_message = ConversationMessage(
        conversation_id=conversation_id,
        role=role,
        content=content,
        metadata_=metadata # Store in the DB field metadata_
    )
    db.add(db_message)

    # Optionally update the conversation's updated_at timestamp
    # await db.execute(
    #     update(Conversation)
    #     .where(Conversation.id == conversation_id)
    #     .values(updated_at=datetime.utcnow())
    # ) # This requires importing 'update' and 'datetime'

    await db.flush()
    await db.refresh(db_message)
    logger.debug(f"Created message ID: {db_message.id}")
    return db_message


async def get_messages_for_conversation(
    db: AsyncSession,
    conversation_id: int,
    limit: int = 10 # Get last N messages for history context
) -> List[ConversationMessage]:
    """Gets the most recent messages for a specific conversation."""
    logger.debug(f"Getting latest {limit} messages for conversation {conversation_id}")
    stmt = (
        select(ConversationMessage)
        .where(ConversationMessage.conversation_id == conversation_id)
        .order_by(desc(ConversationMessage.created_at)) # Get latest first
        .limit(limit)
    )
    result = await db.execute(stmt)
    # Reverse the results to get chronological order (oldest first) for history formatting
    messages = result.scalars().all()[::-1]
    logger.debug(f"Retrieved {len(messages)} messages.")
    return messages

async def get_formatted_history_tuples(
    db: AsyncSession,
    conversation_id: int,
    limit: int = 5 # How many turns (user+ai = 1 turn usually) for prompt
) -> List[Tuple[str, str]]:
     """Gets recent messages and formats them as (role, content) tuples."""
     # Limit messages retrieved (limit * 2 roughly covers user+AI pairs)
     db_messages = await get_messages_for_conversation(db, conversation_id, limit=limit * 2)
     history_tuples = [(msg.role.value, msg.content) for msg in db_messages]
     return history_tuples

# --- Optional: Add functions to get full conversation details, delete conversations, etc. ---
async def get_conversation_with_messages(db: AsyncSession, conversation_id: int) -> Optional[Conversation]:
     """Gets a conversation and eagerly loads its messages."""
     stmt = select(Conversation).where(Conversation.id == conversation_id).options(selectinload(Conversation.messages))
     result = await db.execute(stmt)
     return result.scalars().first()