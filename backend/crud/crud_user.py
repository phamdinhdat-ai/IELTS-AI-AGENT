from typing import Optional, List

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload # If loading relationships

from core.security import get_password_hash, verify_password
from db.models.user import User
from schemas.user import UserCreate, UserUpdate

async def get_user(db: AsyncSession, user_id: int) -> Optional[User]:
    """Gets a single user by ID."""
    result = await db.execute(select(User).filter(User.id == user_id))
    return result.scalars().first()

async def get_user_by_email(db: AsyncSession, email: str) -> Optional[User]:
    """Gets a single user by email."""
    result = await db.execute(select(User).filter(User.email == email))
    return result.scalars().first()

async def get_users(db: AsyncSession, skip: int = 0, limit: int = 100) -> List[User]:
    """Gets a list of users with pagination."""
    result = await db.execute(
        select(User)
        .order_by(User.id)
        .offset(skip)
        .limit(limit)
    )
    return result.scalars().all()

async def create_user(db: AsyncSession, user_in: UserCreate) -> User:
    """Creates a new user."""
    hashed_password = get_password_hash(user_in.password)
    # Create user instance without plain password
    db_user = User(
        email=user_in.email,
        hashed_password=hashed_password,
        full_name=user_in.full_name
        # role is set by default in the model or can be assigned here if needed
    )
    db.add(db_user)
    await db.flush() # Flush to get the ID before returning (or commit happens in get_async_db)
    await db.refresh(db_user)
    return db_user

async def update_user(db: AsyncSession, db_user: User, user_in: UserUpdate) -> User:
    """Updates an existing user."""
    update_data = user_in.model_dump(exclude_unset=True) # Get only provided fields

    if "password" in update_data and update_data["password"]:
        hashed_password = get_password_hash(update_data["password"])
        db_user.hashed_password = hashed_password
        del update_data["password"] # Don't try to set plain password directly

    for field, value in update_data.items():
        setattr(db_user, field, value)

    db.add(db_user)
    await db.flush()
    await db.refresh(db_user)
    return db_user

async def authenticate_user(db: AsyncSession, email: str, password: str) -> Optional[User]:
    """Authenticates a user by email and password."""
    user = await get_user_by_email(db, email=email)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user

async def is_active(user: User) -> bool:
    """Checks if a user is active."""
    return user.is_active # Assuming is_active field exists

async def is_superuser(user: User) -> bool:
    """Checks if a user has admin role."""
    return user.role == "admin" # Assuming role field exists and 'admin' is a role

# Add delete_user function if needed