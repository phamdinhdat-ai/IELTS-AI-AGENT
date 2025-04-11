from fastapi import APIRouter

# Import the endpoint routers
from api.endpoints import auth, chat # Add other endpoint modules here if created (e.g., users, knowledge_admin)

# Create the main API router
api_router = APIRouter()

# Include the authentication routes
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])

# Include the chat routes
api_router.include_router(chat.router, prefix="/chat", tags=["Chat"])

# Example: Include a user management router (if you create endpoints/users.py)
# from app.api.endpoints import users
# api_router.include_router(users.router, prefix="/users", tags=["Users"])

# Example: Include a knowledge base admin router (if you create endpoints/knowledge_admin.py)
# from app.api.endpoints import knowledge_admin
# api_router.include_router(knowledge_admin.router, prefix="/admin/knowledge", tags=["Knowledge Admin"])