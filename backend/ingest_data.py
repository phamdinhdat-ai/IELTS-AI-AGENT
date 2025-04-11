import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any

# --- Add project root to sys.path ---
# This allows importing app modules from the script
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# --- Imports from app ---
from core.config import settings
from db.base import AsyncSessionLocal # Import async session maker
from db.models.knowledge import AccessLevel # Import Enum
from crud import crud_knowledge # Import CRUD function
from schemas.knowledge import KnowledgeChunkCreate # Import Pydantic schema

# --- Langchain Imports ---
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
    # Add more loaders as needed
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Constants ---
PUBLIC_DATA_DIR = project_root / "data_to_ingest" / "public"
INTERNAL_DATA_DIR = project_root / "data_to_ingest" / "internal"
SUPPORTED_LOADERS = {
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".pdf": PyPDFLoader, # Basic PDF loader
    ".docx": UnstructuredWordDocumentLoader,
    # Add more extensions and loaders here
}

# --- Embedding Model Initialization ---
# Load embedding model once
try:
    logger.info(f"Loading embedding model for ingestion: {settings.EMBEDDING_MODEL_NAME}")
    encode_kwargs = {'normalize_embeddings': True}
    embedding_model = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}, # Use 'cuda' if GPU available
        encode_kwargs=encode_kwargs
    )
    logger.info("Embedding model loaded successfully.")
except Exception as e:
    logger.exception(f"Fatal: Failed to load embedding model. Exiting. Error: {e}")
    sys.exit(1) # Exit if embedding model fails

# --- Helper Functions ---

def get_documents_from_directory(directory_path: Path, access_level: AccessLevel) -> List[Dict[str, Any]]:
    """Loads documents from a directory, assigning access level."""
    loaded_documents = []
    logger.info(f"Scanning directory: {directory_path} for access level: {access_level.value}")

    if not directory_path.is_dir():
        logger.warning(f"Directory not found: {directory_path}")
        return []

    for file_path in directory_path.rglob('*'): # Recursively find all files
        if file_path.is_file():
            file_ext = file_path.suffix.lower()
            loader_cls = SUPPORTED_LOADERS.get(file_ext)

            if loader_cls:
                logger.info(f"Loading document: {file_path} using {loader_cls.__name__}")
                try:
                    # Instantiate loader with file path
                    loader = loader_cls(str(file_path))
                    # Load documents (might return multiple docs for some loaders)
                    docs = loader.load()
                    for doc in docs:
                        # Add essential metadata here before chunking
                        doc.metadata["source"] = str(file_path.name) # Filename as source
                        doc.metadata["access_level"] = access_level.value # Assign access level
                        doc.metadata["full_path"] = str(file_path) # Optional: store full path
                    loaded_documents.extend(docs)
                    logger.debug(f"Loaded {len(docs)} document parts from {file_path.name}")
                except Exception as e:
                    logger.error(f"Failed to load document {file_path}: {e}", exc_info=True)
            else:
                logger.warning(f"Skipping unsupported file type: {file_path}")

    logger.info(f"Found {len(loaded_documents)} loadable document parts in {directory_path}")
    return loaded_documents


def chunk_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Splits loaded documents into chunks."""
    logger.info(f"Chunking {len(documents)} document parts...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True, # Optional: Helps track chunk position
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks.")
    return chunks

async def ingest_data():
    """Main function to load, chunk, embed, and store data."""
    logger.info("Starting data ingestion process...")

    # 1. Load Documents with Access Levels
    public_docs = get_documents_from_directory(PUBLIC_DATA_DIR, AccessLevel.public)
    internal_docs = get_documents_from_directory(INTERNAL_DATA_DIR, AccessLevel.internal)
    all_docs = public_docs + internal_docs

    if not all_docs:
        logger.warning("No documents found to ingest. Exiting.")
        return

    # 2. Chunk Documents
    all_chunks = chunk_documents(all_docs)

    # 3. Generate Embeddings (Batching recommended for large datasets)
    logger.info(f"Generating embeddings for {len(all_chunks)} chunks (this may take time)...")
    chunk_contents = [chunk.page_content for chunk in all_chunks]
    try:
        # Use embed_documents for batch processing
        embeddings = await asyncio.to_thread(embedding_model.embed_documents, chunk_contents)
        logger.info("Embeddings generated successfully.")
        if len(embeddings) != len(all_chunks):
             logger.error(f"Mismatch between chunks ({len(all_chunks)}) and embeddings ({len(embeddings)}). Aborting.")
             return
    except Exception as e:
         logger.exception(f"Fatal: Failed during embedding generation. Error: {e}")
         return # Stop ingestion if embedding fails

    # 4. Prepare Data for Database Insertion
    chunks_to_create: List[KnowledgeChunkCreate] = []
    processed_doc_ids = set() # Keep track of processed document IDs

    for i, chunk in enumerate(all_chunks):
        doc_id = chunk.metadata.get("source", f"unknown_doc_{i}")
        processed_doc_ids.add(doc_id) # Add filename to set of processed documents

        # Ensure metadata has required fields
        metadata_for_db = {
            "source": doc_id,
            "access_level": chunk.metadata.get("access_level", AccessLevel.public.value), # Default safety
            "page": chunk.metadata.get("page"), # From PyPDFLoader etc.
            "start_index": chunk.metadata.get("start_index"), # From splitter
            # Add any other relevant metadata extracted during loading
        }

        chunk_data = KnowledgeChunkCreate(
            document_id=doc_id,
            chunk_number=i, # Simple index, could be more sophisticated
            content=chunk.page_content,
            embedding=embeddings[i],
            metadata_=metadata_for_db
        )
        chunks_to_create.append(chunk_data)

    # 5. Interact with Database (using async session)
    logger.info(f"Preparing to ingest {len(chunks_to_create)} chunks into the database...")
    async with AsyncSessionLocal() as session:
        async with session.begin(): # Start a transaction
            # --- Optional: Clean up old chunks for processed documents ---
            if processed_doc_ids:
                logger.info(f"Deleting existing chunks for {len(processed_doc_ids)} processed document IDs...")
                # Note: This deletes ALL chunks for a doc ID even if only some chunks changed.
                # More sophisticated updates require tracking individual chunk changes.
                for doc_id_to_delete in processed_doc_ids:
                    await crud_knowledge.delete_knowledge_chunks_by_doc_id(session, doc_id_to_delete)
                logger.info("Deletion of old chunks complete.")

            # --- Insert new chunks ---
            logger.info(f"Inserting {len(chunks_to_create)} new chunks...")
            inserted_count = 0
            for chunk_in in chunks_to_create:
                try:
                    await crud_knowledge.create_knowledge_chunk(session, chunk_in=chunk_in)
                    inserted_count += 1
                except Exception as e:
                    logger.error(f"Failed to insert chunk for doc {chunk_in.document_id}, chunk approx {chunk_in.chunk_number}: {e}", exc_info=False) # Log error but continue? Or raise/rollback?
            logger.info(f"Successfully inserted {inserted_count} chunks.")

            # Transaction commits automatically if no exceptions raised within `async with session.begin():`
        logger.info("Database transaction committed.")

    logger.info("Data ingestion process finished.")


if __name__ == "__main__":
    # Ensure data directories exist (optional check)
    if not PUBLIC_DATA_DIR.exists():
        logger.warning(f"Public data directory not found: {PUBLIC_DATA_DIR}")
    if not INTERNAL_DATA_DIR.exists():
        logger.warning(f"Internal data directory not found: {INTERNAL_DATA_DIR}")

    # Run the async ingestion function
    asyncio.run(ingest_data())