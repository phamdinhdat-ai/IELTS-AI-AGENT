from pydantic import BaseModel, EmailStr, ConfigDict
from typing import Optional
from datetime import datetime

from db.models.user import UserRole # Import the Enum

# --- Base Schemas ---
class UserBase(BaseModel):
    email: EmailStr # Use EmailStr for validation
    full_name: Optional[str] = None

# --- Properties to receive via API on creation ---
class UserCreate(UserBase):
    password: str # Receive plain password on creation

# --- Properties to receive via API on update ---
class UserUpdate(UserBase):
    password: Optional[str] = None # Optional password update
    full_name: Optional[str] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None

# --- Properties stored in DB (never directly expose password) ---
class UserInDBBase(UserBase):
    id: int
    role: UserRole
    is_active: bool
    created_at: datetime
    updated_at: datetime
    model_config = ConfigDict(from_attributes=True) # Allow mapping from ORM model

# --- Additional properties stored in DB ---
# Separate class used internally, includes hashed password
class UserInDB(UserInDBBase):
    hashed_password: str

# --- Properties to return to client ---
class User(UserInDBBase):
    # Excludes hashed_password by inheriting from UserInDBBase
    pass