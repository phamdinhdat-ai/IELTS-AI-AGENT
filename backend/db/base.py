from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import MetaData

from core.config import settings

# Create the async engine
# pool_recycle=3600 helps prevent connection drops on idle connections
engine = create_async_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    echo=False # Set to True for debugging SQL queries
    # pool_recycle=3600 # Optional: Recycle connections periodically
)

# Create the async session maker
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False # Important for async usage
)

# Define metadata with naming convention for constraints (optional but good practice)
convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}
metadata = MetaData(naming_convention=convention)

# Define the Base class for declarative models
class Base(DeclarativeBase):
    metadata = metadata

# Dependency function to get a DB session (to be used in API endpoints)
async def get_async_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit() # Commit transaction if everything succeeds
        except Exception:
            await session.rollback() # Rollback on error
            raise
        finally:
            await session.close() # Ensure session is closed