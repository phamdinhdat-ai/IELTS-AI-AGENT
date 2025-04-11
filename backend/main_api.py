import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles # If serving frontend from FastAPI

from core.config import settings
from api.api import api_router # Import the main router
# Optional: Import model loading functions if using lifespan
# from app.services.rag_service import load_models_on_startup, close_models_on_shutdown

# --- Logging Setup ---
# Configure logging (can be more sophisticated using logging.config)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Optional: Lifespan for loading models ---
# This is the recommended way to load heavy resources like AI models
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Code to run on startup
#     logger.info("Application startup: Loading resources...")
#     # Load models, initialize DB connections pools etc.
#     # app.state.embedding_model = load_embedding_model() # Store models in app state
#     # app.state.llm = load_llm()
#     # load_models_on_startup() # Call function from rag_service or elsewhere
#     logger.info("Resources loaded.")
#     yield
#     # Code to run on shutdown
#     logger.info("Application shutdown: Cleaning up resources...")
#     # close_models_on_shutdown() # Clean up resources if needed
#     logger.info("Cleanup finished.")

# --- Create FastAPI App ---
# Pass lifespan context manager if using it: lifespan=lifespan
app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json", # Standard OpenAPI spec URL
    version="0.1.0" # Example version
    # lifespan=lifespan # Uncomment if using lifespan events
)

# --- CORS Middleware ---
# Set all CORS enabled origins
if settings.BACKEND_CORS_ORIGINS:
    logger.info(f"Configuring CORS for origins: {settings.BACKEND_CORS_ORIGINS}")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS], # Convert Pydantic AnyHttpUrl back to str if needed
        allow_credentials=True,
        allow_methods=["*"], # Allow all methods
        allow_headers=["*"], # Allow all headers
    )
else:
     logger.warning("CORS origins not configured. Allowing all origins (development mode only).")
     # Allow all origins if none are specified (useful for local dev)
     app.add_middleware(
         CORSMiddleware,
         allow_origins=["*"],
         allow_credentials=True,
         allow_methods=["*"],
         allow_headers=["*"],
     )

# --- Include API Router ---
# Include the main router with the configured API prefix
app.include_router(api_router, prefix=settings.API_V1_STR)
logger.info(f"API router included at prefix: {settings.API_V1_STR}")


# --- Root Endpoint (Optional) ---
@app.get("/", tags=["Root"])
async def read_root():
    """Simple health check or welcome message."""
    return {"message": f"Welcome to {settings.PROJECT_NAME}"}

# --- Static Files (Optional - Serve Frontend) ---
# Mount this LAST if you want '/' to serve index.html
# Ensure the path 'static' exists at the root level where you run uvicorn/docker
# try:
#     app.mount("/", StaticFiles(directory="static", html=True), name="static")
#     logger.info("Static files mounted for serving frontend.")
# except RuntimeError as e:
#     logger.warning(f"Could not mount static files directory: {e}. Frontend will not be served by FastAPI.")


# Note: __main__ block for running with uvicorn directly is removed
# as it's better practice to run via `uvicorn app.main:app` or Docker CMD.