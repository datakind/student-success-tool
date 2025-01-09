"""Main file for the SST Worker.
"""

import logging
from typing import Any, Annotated
from fastapi import FastAPI
from fastapi.responses import FileResponse

from src.webapp.config import env_vars, startup_env_vars
from src.webapp.database import setup_db, db_engine
from pydantic import BaseModel
import pandas as pd
from google.cloud import storage


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


@app.on_event("startup")
def on_startup():
    print("Starting up app...")
    startup_env_vars()
    setup_db(env_vars["ENV"])


# On shutdown, we have to cleanup the GCP database connections
@app.on_event("shutdown")
async def shutdown_event():
    print("Performing shutdown tasks...")
    await db_engine.dispose()


# The following root paths don't have pre-authn.
@app.get("/")
def read_root() -> Any:
    """Returns the index.html file."""
    return FileResponse("src/worker/index.html")


class DataUploadValidationRequest(BaseModel):
    filename: str
    inst_id: str


@app.post("/validate-data-upload")
def validate_file(request: DataUploadValidationRequest) -> Any:
    """Validates the file."""
    client = storage.Client()
    bucket = client.bucket(request.inst_id)
    blob = bucket.blob(f"unvalidated/{request.filename}")
    logger.info(f"Blob content type: {blob.content_type}")
    return {
        "status": "ok",
        "filename": request.filename,
        "inst_id": request.inst_id,
        "content_type": blob.content_type,
    }
