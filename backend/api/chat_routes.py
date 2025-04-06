from fastapi import APIRouter, HTTPException # Import HTTPException
from pydantic import BaseModel
from models import ChatRequest, ChatResponse  # Importing request/response models

# --- Import the NLP processor ---
# Make sure the ai_models directory is in the Python path or use relative import
try:
    from ai_models import nlp_processor
except ImportError:
    # Handle case where file/directory structure isn't quite right yet
    print("Could not import nlp_processor. Make sure ai_models/nlp_processor.py exists.")
    # Define a dummy function if import fails, so the API can still run initially
    class nlp_processor:
        @staticmethod
        def process_text(text: str): return {"error": "NLP processor not available"}


# --- Router Setup ---
router = APIRouter(prefix="/chat", tags=["Chat"])


# --- Chat Endpoint ---
@router.post("/", response_model=ChatResponse)
async def handle_chat(request: ChatRequest):
    """
    Receives a user message, processes it with basic NLP, and returns a reply.
    """
    print(f"Received message via chat router: {request.message}")

    # --- Call the NLP Processor ---
    nlp_results = nlp_processor.process_text(request.message)

    # --- Handle potential NLP errors ---
    if "error" in nlp_results:
        # Log the error server-side
        print(f"NLP processing error: {nlp_results['error']}")
        # Raise an HTTPException to inform the client something went wrong
        raise HTTPException(status_code=500, detail="Error processing message internally.")
        # Or, return a user-friendly error message in the reply:
        # return ChatResponse(reply=f"Sorry, there was an error processing your message.")

    # --- Construct Reply using NLP results ---
    num_tokens = nlp_results.get("num_tokens", 0)
    tokens_str = ", ".join(nlp_results.get("tokens", []))

    reply_text = f"Processed your message. It has {num_tokens} tokens. Tokens found: [{tokens_str}]"

    # --- Placeholder for Core Logic ---
    # Next, you would pass nlp_results (or parts of it) to your
    # core/chat_manager.py or core/ielts_logic.py to decide the *actual* reply
    # based on recognized intent, entities, and conversation state.

    return ChatResponse(reply=reply_text) #, nlp_results=nlp_results) # Optionally return NLP results