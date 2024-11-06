# This Dockerfile is is not suitable for production use, but is useful for
# development and testing. A production Dockerfile would likely use a
# multi-stage build to separate the build environment from the runtime.

# Install uv
FROM python:3.10-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Change the working directory to the `app` directory
WORKDIR /app

ADD uv.lock pyproject.toml /app/

# Install dependencies
RUN uv sync --frozen --no-install-project

# Copy the project into the image
ADD . /app

# Sync the project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

# Run the application.
CMD ["fastapi", "run", "src/webapp", "--port", "8080", "--host", "0.0.0.0"]