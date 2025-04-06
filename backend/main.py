# In ielts_chatbot_backend/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn # Required to run the app using uvicorn
from models import ChatRequest, ChatResponse # Importing request/response models
# from api import chat_routes, speech_routes # Importing routers from the api directory
from api import chat_routes, speech_routes # Importing routers from the api directory
# --- Initialize FastAPI App ---
app = FastAPI(
    title="IELTS Chatbot API",
    description="API endpoints for the IELTS learning chatbot.",
    version="0.1.0"
)
# --- Include Routers from the api directory ---
# This incorporates all routes defined in chat_routes.py and speech_routes.py
app.include_router(chat_routes.router)
app.include_router(speech_routes.router)
# --- CORS (Cross-Origin Resource Sharing) Middleware ---
# Allows your frontend (potentially running on a different port/domain)
# to communicate with this backend.
# Adjust origins as needed for production.
origins = [
    "http://localhost",         # Allow localhost (common for development)
    "http://localhost:8080",    # Example frontend dev port
    "http://localhost:3000",    # Example frontend dev port (React)
    "http://127.0.0.1",
    "http://127.0.0.1:8080",
    "http://127.0.0.1:3000",
    # Add your deployed frontend URL here when ready
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods (GET, POST, etc.)
    allow_headers=["*"], # Allow all headers
)


# --- A Simple Test Route ---
@app.get("/ping", tags=["Health Check"])
async def ping():
    """
    A simple endpoint to check if the API is running.
    """
    return {"message": "Backend server is running with FastAPI!"}

# --- Include Routers (We'll add these later) ---
# from api import chat_routes, speech_routes
# app.include_router(chat_routes.router)
# app.include_router(speech_routes.router)

# --- Main Execution (for running with `python main.py`, less common for FastAPI) ---
# It's more standard to run FastAPI with Uvicorn directly from the command line.
# However, this block allows running `python main.py` for simple testing if needed.
if __name__ == "__main__":
    # Uvicorn is the recommended ASGI server for FastAPI
    # Run from terminal: uvicorn main:app --reload
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)