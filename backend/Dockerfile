# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies that might be needed by libraries (e.g., for unstructured)
# Example: RUN apt-get update && apt-get install -y --no-install-recommends some-package && rm -rf /var/lib/apt/lists/*
# Add dependencies needed by unstructured, PyPDF2, etc. if required

# Copy the requirements file into the container at /app
COPY ./requirements.txt /app/

# Install any needed packages specified in requirements.txt
# Use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY . /app/

# Make port 8000 available to the world outside this container
EXPOSE 8765

# Define environment variables (can be overridden)
# Ensure these match the ones used in main.py if not using a .env file inside container
ENV API_HOST="0.0.0.0"
ENV API_PORT=8765
ENV PERSIST_DIRECTORY="/app/chroma_db_persist" 
ENV DATA_PATH="/app/data" 
# Add other ENV vars as needed (OLLAMA_BASE_URL, etc.)

# Run main.py when the container launches
# Use "CMD" for default command or "ENTRYPOINT"
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8765"]