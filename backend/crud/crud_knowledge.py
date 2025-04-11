import logging
from typing import List, Optional, Sequence, Union

from sqlalchemy import select, delete, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import JSONB # For JSONB operations

# Import pgvector operators for similarity search
# Choose one based on your index/preference (L2/IP/Cosine)
from pgvector.sqlalchemy import Vector, l2_distance, max_inner_product, cosine_distance

from db.models.knowledge import KnowledgeChunk, AccessLevel
from db.models.user import UserRole
from schemas.knowledge import KnowledgeChunkCreate # Used for ingestion

logger = logging.getLogger(__name__)

async def create_knowledge_chunk(db: AsyncSession, chunk_in: KnowledgeChunkCreate) -> KnowledgeChunk:
    """Creates a new knowledge chunk in the database."""
    db_chunk = KnowledgeChunk(
        document_id=chunk_in.document_id,
        chunk_number=chunk_in.chunk_number,
        content=chunk_in.content,
        embedding=chunk_in.embedding, # Assumes embedding is List[float]
        metadata_=chunk_in.metadata_ # Use metadata_ field
    )
    db.add(db_chunk)
    await db.flush()
    await db.refresh(db_chunk)
    logger.debug(f"Created Knowledge Chunk ID: {db_chunk.id}")
    return db_chunk

async def get_knowledge_chunk(db: AsyncSession, chunk_id: int) -> Optional[KnowledgeChunk]:
    """Gets a single knowledge chunk by its ID."""
    result = await db.execute(select(KnowledgeChunk).filter(KnowledgeChunk.id == chunk_id))
    return result.scalars().first()

async def delete_knowledge_chunks_by_doc_id(db: AsyncSession, document_id: str) -> int:
    """Deletes all chunks associated with a specific document ID. Returns count deleted."""
    stmt = delete(KnowledgeChunk).where(KnowledgeChunk.document_id == document_id)
    result = await db.execute(stmt)
    deleted_count = result.rowcount
    logger.info(f"Deleted {deleted_count} chunks for document_id: {document_id}")
    return deleted_count

async def get_relevant_chunks_for_user(
    db: AsyncSession,
    embedding: List[float],
    user_role: UserRole,
    limit: int = 3,
    min_similarity: Optional[float] = None # Optional threshold
) -> Sequence[KnowledgeChunk]:
    """
    Finds relevant knowledge chunks based on embedding similarity and user role authorization.

    Args:
        db: Async SQLAlchemy session.
        embedding: The query embedding vector (List[float]).
        user_role: The role of the user querying (UserRole Enum).
        limit: Max number of chunks to return.
        min_similarity: Optional minimum cosine similarity score (0 to 1).
                        Note: pgvector distance operators return distance (0=identical for cosine).

    Returns:
        A list of KnowledgeChunk objects, potentially with a 'score' attribute added.
    """
    logger.debug(f"Searching for relevant chunks for role: {user_role}, limit: {limit}")

    # Determine accessible levels based on role
    allowed_access_levels: List[str]
    if user_role in [UserRole.employee, UserRole.admin]:
        allowed_access_levels = [AccessLevel.public.value, AccessLevel.internal.value]
    else: # Customer
        allowed_access_levels = [AccessLevel.public.value]

    # --- Build the WHERE clause for authorization ---
    # Check if the 'access_level' key in the JSONB metadata_ field is in the allowed list
    # Requires jsonb_path_exists or similar depending on PG version and indexing strategy
    # Simpler version using ->> text operator (less performant without specific index)
    authorization_filter = KnowledgeChunk.metadata_['access_level'].astext.in_(allowed_access_levels)

    # --- Choose the distance function ---
    # Cosine Distance: 0 = identical, 2 = opposite. Lower is better.
    # We want chunks with distance <= threshold if using min_similarity
    distance_func = KnowledgeChunk.embedding.cosine_distance(embedding)
    # Alternative: Max Inner Product (for normalized embeddings, higher is better)
    # distance_func = KnowledgeChunk.embedding.max_inner_product(embedding) * -1 # Negate to order ascending (lower is better)
    # Alternative: L2 Distance (lower is better)
    # distance_func = KnowledgeChunk.embedding.l2_distance(embedding)

    # --- Build the base query ---
    stmt = select(
        KnowledgeChunk,
        distance_func.label("distance") # Calculate distance and label it
    ).where(authorization_filter)

    # --- Add similarity threshold if provided ---
    if min_similarity is not None:
        # Convert similarity threshold (0-1) to distance threshold (1-0 for cosine)
        # similarity = 1 - distance --> distance = 1 - similarity
        distance_threshold = 1.0 - min_similarity
        stmt = stmt.where(distance_func <= distance_threshold) # Filter by distance

    # --- Order by distance and limit results ---
    stmt = stmt.order_by(distance_func).limit(limit)

    # --- Execute query ---
    logger.debug(f"Executing vector search query with filter: {authorization_filter}")
    results = await db.execute(stmt)
    chunks_with_distance = results.all() # Returns tuples of (KnowledgeChunk, distance)

    # --- Process results (add score) ---
    processed_chunks = []
    for chunk, distance in chunks_with_distance:
        # Calculate cosine similarity from distance: similarity = 1 - distance
        chunk.score = 1.0 - distance if distance is not None else 0.0
        processed_chunks.append(chunk)
        logger.debug(f"Retrieved chunk ID {chunk.id}, Distance: {distance:.4f}, Similarity: {chunk.score:.4f}")

    logger.info(f"Retrieved {len(processed_chunks)} relevant chunks for role {user_role}.")
    return processed_chunks

# Add other CRUD functions if needed, e.g., updating chunks (rare for RAG)