# This Dockerfile utilizes a multi-stage build to separate the build environment from the runtime.

# Stage 1: Build stage
FROM python:3.10-slim-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y build-essential

# Set the working directory
WORKDIR /app

# Copy the dependency files
COPY uv.lock pyproject.toml /app/

# Install dependencies
RUN pip install uv && uv sync --frozen --no-install-project

# Stage 2: Production stage
FROM python:3.10-slim-bookworm

# Copy uv from the builder stage
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory
WORKDIR /app

# Copy only the necessary files from the builder stage
COPY --from=builder /app /app

# Copy the project files
COPY . /app

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"

# Expose the port
EXPOSE 8080

# Run the application
CMD ["fastapi", "run", "src/webapp", "--host", "0.0.0.0", "--port", "8080"]
