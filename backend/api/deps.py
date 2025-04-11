from typing import Generator, Optional, Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from core import security
from core.config import settings
from db.base import get_async_db # Import the async session getter
from db.models.user import User, UserRole
from crud import crud_user
from schemas.token import TokenPayload # Import if validating payload structure

# Define the OAuth2 scheme pointing to the token URL (login endpoint)
# This tells FastAPI how to extract the token (from Authorization: Bearer header)
reusable_oauth2 = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_STR}/auth/login"
)

# Type hint for dependency injection
SessionDep = Annotated[AsyncSession, Depends(get_async_db)]
TokenDep = Annotated[str, Depends(reusable_oauth2)]

async def get_current_user(
    db: SessionDep, token: TokenDep
) -> User:
    """
    Dependency to get the current authenticated user from the token.
    Raises HTTPException 401 if authentication fails.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    subject = security.verify_access_token(token)
    if subject is None:
        raise credentials_exception

    # Assuming the subject ('sub' claim) is the user's email
    user = await crud_user.get_user_by_email(db, email=subject)
    if user is None:
        raise credentials_exception

    # Optional: Check if token payload matches a specific schema
    # try:
    #     token_data = TokenPayload(sub=email)
    # except ValidationError:
    #     raise credentials_exception

    return user

# Dependency to get the current active user
async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)]
) -> User:
    """Checks if the current user is active."""
    if not crud_user.is_active(current_user):
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# Dependency to get the current active employee or admin
async def get_current_active_employee(
    current_user: Annotated[User, Depends(get_current_active_user)]
) -> User:
    """Checks if the current user is an employee or admin."""
    if current_user.role not in [UserRole.employee, UserRole.admin]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="The user doesn't have enough privileges (requires employee or admin)",
        )
    return current_user

# Dependency to get the current active superuser (admin)
async def get_current_active_superuser(
    current_user: Annotated[User, Depends(get_current_active_user)]
) -> User:
    """Checks if the current user is an admin."""
    if current_user.role != UserRole.admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="The user doesn't have enough privileges (requires admin)",
        )
    return current_user

# Define common dependency types for easier use in endpoints
CurrentUser = Annotated[User, Depends(get_current_active_user)]
CurrentEmployee = Annotated[User, Depends(get_current_active_employee)]
CurrentAdmin = Annotated[User, Depends(get_current_active_superuser)]