"""Main file for the SST Worker.
"""

import logging
from typing import Any, Annotated
from fastapi import FastAPI
from fastapi.responses import FileResponse

from src.webapp.config import env_vars, startup_env_vars
from src.webapp.database import setup_db, db_engine
from src.webapp.validation import validate_file_reader
from src.webapp.gcsutil import rename_file
from pydantic import BaseModel
import pandas as pd
from google.cloud import storage
from fastapi import HTTPException


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


class SendNotificationRequest(BaseModel):
    inst_id: str
    roles: list[str]
    content: str


@app.post("/send-notification")
def send_notification(request: SendNotificationRequest) -> Any:
    """Sends a notification."""
    return {"content": "Notification sent."}


class DataUploadValidationRequest(BaseModel):
    protoPayload: Annotated[dict, "The protoPayload object."]


@app.post("/validate-data-upload")
def validate_file(request: DataUploadValidationRequest) -> Any:
    """Validates the file."""
    resource = request.protoPayload.get("resourceName")
    if not resource:
        raise HTTPException(status_code=422, detail="resourceName is required.")
    parts = resource.split("/")
    inst_id, filename = parts[3], parts[-1]
    if not inst_id or not filename:
        raise HTTPException(
            status_code=422,
            detail="resourceName must be in the format: projects/.../buckets/.../objects/...",
        )
    client = storage.Client()
    bucket = client.bucket(inst_id)
    blob = bucket.blob(f"unvalidated/{filename}")
    new_blob_name = f"validated/{filename}"
    with blob.open("r") as file:
        try:
            with blob.open("r") as file:
                validate_file_reader(file)
        except Exception as e:
            logger.error(f"Error validating file: {e}")
            new_blob_name = f"invalid/{filename}"
    rename_and_move_blob(bucket, blob, new_blob_name)
    return {
        "filename": new_blob_name,
        "inst_id": inst_id,
        "content_type": blob.content_type,
    }


def rename_and_move_blob(bucket, blob, new_blob_name):
    logger.debug(f"Renaming file to: {new_blob_name}")
    bucket.copy_blob(blob, bucket, new_blob_name)
    blob.delete()
    logger.debug(f"File renamed to: {new_blob_name}")
