# Stage 1: Build base with Python and Poetry/Pip
FROM python:3.10-slim as python-base
ENV PYTHONUNBUFFERED=1 \
    # prevents python creating .pyc files
    PYTHONDONTWRITEBYTECODE=1 \
    \
    # pip
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    \
    # poetry
    # POETRY_VERSION=1.7.1 \
    # POETRY_HOME="/opt/poetry" \
    # POETRY_NO_INTERACTION=1 \
    \
    # paths
    PATH="$PATH:/root/.local/bin"
    # PATH="$POETRY_HOME/bin:$PATH"

# System dependencies (add as needed by your Python packages)
# Example: build-essential for packages needing compilation, git, curl, etc.
# Example: libpq-dev for psycopg
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        build-essential \
        libpq-dev \
        # Add other system dependencies required by unstructured, etc.
        # Example: libreoffice tesseract-ocr poppler-utils
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install pip requirements
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Final application image
FROM python:3.10-slim as final

WORKDIR /app

# Copy installed packages from the previous stage
COPY --from=python-base /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=python-base /root/.local /root/.local

# Copy application code
COPY ./app /app/app
COPY ./alembic /app/alembic
COPY alembic.ini /app/alembic.ini
# Copy .env ONLY if absolutely necessary and you understand the security implications
# It's better to inject environment variables via Docker Compose or orchestrator
# COPY .env /app/.env

# Set non-root user for security (optional but recommended)
RUN useradd --create-home --shell /bin/bash appuser
USER appuser
WORKDIR /home/appuser/app # Adjust WORKDIR if using non-root user

# Expose the port the app runs on
EXPOSE 8000

# --- Entrypoint Script (Optional but Recommended) ---
# Create an entrypoint script (e.g., entrypoint.sh) to run migrations before starting the app
# COPY entrypoint.sh /app/entrypoint.sh
# RUN chmod +x /app/entrypoint.sh
# ENTRYPOINT ["/app/entrypoint.sh"]

# --- Default Command (if not using entrypoint script) ---
# Run migrations first, then start Uvicorn. Use '&&' to ensure migration finishes.
# Note: Running migrations at startup might slow down scaling/restarts.
# Consider running migrations as a separate step/job in production.
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]

# If using an entrypoint.sh:
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]