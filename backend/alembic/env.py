import os
import sys
from logging.config import fileConfig
from sqlalchemy.ext.asyncio import create_async_engine # Use async engine
from sqlalchemy import pool # Import pool for NullPool

from alembic import context

# --- Append app directory to sys.path ---
# This allows Alembic to find your application modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# --- Import your app's Base and settings ---
from db.base import Base # Your SQLAlchemy declarative base
from core.config import settings # Your application settings
# Import all models here so Base metadata knows about them
from db.models import user, knowledge, chat # Add all model modules

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# --- Configure target metadata ---
# Set the SQLAlchemy Base's metadata as the target for autogenerate
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def get_url():
    """Retrieve database URL from settings."""
    return settings.DATABASE_URL

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well. By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True, # Recommended for detecting type changes
        render_as_batch=True, # Enable batch mode for SQLite compatibility (good practice)
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    """Helper function to run migrations using a database connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        render_as_batch=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    # Create an async engine using the URL from settings
    connectable = create_async_engine(
        get_url(),
        poolclass=pool.NullPool, # Use NullPool for migration engine
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations) # Run sync migration logic within async context

    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    # Use asyncio.run to execute the async online migration function
    import asyncio
    asyncio.run(run_migrations_online())