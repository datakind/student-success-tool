"""Main file for the SST API.
"""

from typing import Any
from fastapi import FastAPI
from fastapi.responses import FileResponse

# import logging

from .routers import models, users, data, institutions

app = FastAPI(
    servers=[
        # TODO: placeholders
        {"url": "https://stag.example.com", "description": "Staging environment"},
        {"url": "https://prod.example.com", "description": "Production environment"},
    ],
    root_path="/api/v1",
)

app.include_router(institutions.router)
app.include_router(models.router)
app.include_router(users.router)
app.include_router(data.router)


@app.get("/")
def read_root() -> Any:
    """Returns the index.html file."""
    return FileResponse("index.html")
