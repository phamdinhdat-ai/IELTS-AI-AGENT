from fastapi import APIRouter, Depends, HTTPException, status, Body
from fastapi.security import OAuth2PasswordRequestForm # Standard form for username/password
from typing import Annotated # For Depends

from core import security
from crud import crud_user
from schemas import user as user_schema # Alias to avoid naming clash
from schemas import token as token_schema
from api import deps # Import dependencies

router = APIRouter()

@router.post("/login", response_model=token_schema.Token, summary="User Login")
async def login_for_access_token(
    session: deps.SessionDep, # Use SessionDep type hint
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()]
) -> token_schema.Token:
    """
    OAuth2 compatible token login, get an access token for future requests.
    Uses form data (username/password).
    """
    user = await crud_user.authenticate_user(
        session, email=form_data.username, password=form_data.password # OAuth form uses 'username' for email
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
         raise HTTPException(status_code=400, detail="Inactive user")

    access_token = security.create_access_token(subject=user.email) # Use email as subject
    return token_schema.Token(access_token=access_token, token_type="bearer")

@router.post("/register", response_model=user_schema.User, status_code=status.HTTP_201_CREATED, summary="Register New User")
async def register_new_user(
    session: deps.SessionDep,
    user_in: user_schema.UserCreate = Body(...) # Expect user data in request body
) -> user_schema.User:
    """
    Create new user. Basic registration.
    """
    existing_user = await crud_user.get_user_by_email(session, email=user_in.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The user with this email already exists in the system.",
        )
    user = await crud_user.create_user(session, user_in=user_in)
    # Note: Depending on policy, you might want to assign roles differently,
    # or require email verification etc. This is a basic implementation.
    return user

@router.get("/me", response_model=user_schema.User, summary="Get Current User Info")
async def read_users_me(
    current_user: deps.CurrentUser # Use CurrentUser dependency
) -> user_schema.User:
    """
    Fetch the current logged-in user's information.
    """
    return current_user

# Optional: Add endpoints for password recovery, user updates by admin, etc.