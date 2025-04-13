# user_db.py
from typing import Dict, Optional
from pydantic import BaseModel, Field
from security import get_password_hash # Import hashing function

# --- Pydantic Models for User Data ---
class UserBase(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    customer_id: str # Link user to a customer config

class UserCreate(UserBase):
    password: str

class UserInDBBase(UserBase):
    user_id: int = Field(..., alias='id') # Simulate DB ID

    class Config:
        from_attributes = True # For compatibility if using ORM later
        populate_by_name=True

class UserInDB(UserInDBBase):
    hashed_password: str

# --- Simulated Database (Replace with actual DB interaction) ---
# In-memory dictionary {username: UserInDB object}
fake_users_db: Dict[str, UserInDB] = {}

# --- Pre-populate with some dummy data ---
# Important: Hash passwords before storing!
hashed_pw_alice = get_password_hash("password_a")
fake_users_db["alice_a"] = UserInDB(
    id=1,
    username="alice_a",
    full_name="Alice (Customer A)",
    email="alice@customera.com",
    hashed_password=hashed_pw_alice,
    customer_id="customer_a" # Alice belongs to Customer A
)

hashed_pw_bob = get_password_hash("password_b")
fake_users_db["bob_b"] = UserInDB(
    id=2,
    username="bob_b",
    full_name="Bob (Customer B)",
    email="bob@customerb.com",
    hashed_password=hashed_pw_bob,
    customer_id="customer_b" # Bob belongs to Customer B
)
# --- End Dummy Data ---


def get_user(username: str) -> Optional[UserInDB]:
    """Retrieves a user by username from the fake DB."""
    if username in fake_users_db:
        return fake_users_db[username]
    return None

def add_user(user_data: UserCreate) -> UserInDB:
    """Adds a new user to the fake DB."""
    if user_data.username in fake_users_db:
        raise ValueError(f"Username '{user_data.username}' already exists.")

    hashed_password = get_password_hash(user_data.password)
    user_id = max(user.user_id for user in fake_users_db.values() or [0]) + 1
    user_in_db = UserInDB(
        id=user_id,
        hashed_password=hashed_password,
        **user_data.model_dump(exclude={"password"}) # Don't store plain password
    )
    fake_users_db[user_data.username] = user_in_db
    return user_in_db

# Add more functions as needed (update_user, delete_user, etc.)