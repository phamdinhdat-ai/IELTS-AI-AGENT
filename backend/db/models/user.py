import enum
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, DateTime, Enum as SQLEnum
from sqlalchemy.orm import relationship # If adding relationships later
from sqlalchemy import Boolean
from db.base import Base

UTC = timezone.utc
# Define user roles using Python Enum
class UserRole(str, enum.Enum):
    customer = "customer"
    employee = "employee"
    admin = "admin" # Example: Admins might manage users or data

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, index=True) # Optional: Add more user details
    role = Column(SQLEnum(UserRole), default=UserRole.customer, nullable=False, index=True)
    is_active = Column(Boolean(), default=True) # Optional: Soft delete or deactivation
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Add relationships here if needed, e.g., to conversations
    # conversations = relationship("Conversation", back_populates="user")

    def __repr__(self):
        return f"<User(id={self.id}, email='{self.email}', role='{self.role}')>"