"""Main file for the SST API.
"""

import logging
from typing import Any
from fastapi import FastAPI
from fastapi.responses import FileResponse

from .routers import models, users, data, institutions
from .database import setup_db, db_engine
from .config import env_vars

# Set the logging
logging.basicConfig(format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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


@app.on_event("startup")
def on_startup():
    print("Starting up app...")
    env = env_vars["ENV"]
    if not env:
        raise ValueError(
            "Missing ENV environment variable. Required. Can be PROD, STAGING, or DEV."
        )
    if env not in ["PROD", "STAGING", "DEV"]:
        raise ValueError("ENV environment variable not one of: PROD, STAGING, or DEV.")
    setup_db()


# On shutdown, we have to cleanup the GCP database connections
@app.on_event("shutdown")
async def shutdown_event():
    print("Performing shutdown tasks...")
    await db_engine.dispose()


@app.get("/")
def read_root() -> Any:
    """Returns the index.html file."""
    return FileResponse("src/webapp/index.html")
