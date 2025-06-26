# This Dockerfile uses a multi-stage build to separate the build environment from the runtime.

# Build stage
FROM python:3.10-slim-bookworm AS builder
COPY --from=ghcr.io/astral-sh/uv:0.5.30 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1
ENV UV_NO_CACHE=1

# Change the working directory to the `app` directory
WORKDIR /app

# Add project files
ADD uv.lock pyproject.toml /app/

# Install dependencies
RUN uv sync --frozen --no-install-project

# Copy the project into the image
ADD . /app

# Sync the project
RUN uv sync --frozen --no-editable

# Runtime stage
FROM python:3.10-slim-bookworm

# Copy the environment, but not the source code
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src

ENV PATH="/app/.venv/bin:$PATH"

# Run the application.
CMD ["sh", "-c", "uvicorn src.webapp.main:app --host 0.0.0.0 --port ${PORT}"]
#CMD ["uvicorn", "src.webapp.main:app", "--host", "0.0.0.0", "--port", "8080"]
