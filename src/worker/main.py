"""Main file for the SST Worker.
"""

import logging
from typing import Any, Annotated
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import FileResponse

from pydantic import BaseModel
from google.cloud import storage
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
import jwt
from .utilities import (
    get_sftp_bucket_name,
    StorageControl,
)
from .config import sftp_vars, env_vars, startup_env_vars
from .authn import Token, get_current_username, check_creds, create_access_token
from datetime import timedelta, datetime, timezone

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
    root_path="/worker/api/v1",
)

# this uses api key to auth to backend api, but credentials to auth to this service


class PdpPullRequest(BaseModel):
    """Params for the PDP pull request."""

    placeholder: str | None = None


class PdpPullResponse(BaseModel):
    """Fields for the PDP pull response."""

    pdp_inst_generated: list[int]
    pdp_inst_not_found: list[int]


@app.on_event("startup")
def on_startup():
    print("Starting up app...")
    startup_env_vars()


# On shutdown, we have to cleanup the GCP database connections
@app.on_event("shutdown")
def shutdown_event():
    print("Performing shutdown tasks...")


# The following root paths don't have pre-authn.
@app.get("/")
def read_root() -> Any:
    """Returns the index.html file."""
    return FileResponse("src/worker/index.html")


@app.post("/token")
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
) -> Token:
    valid = check_creds(form_data.username, form_data.password)
    if not valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(
        minutes=int(env_vars["ACCESS_TOKEN_EXPIRE_MINUTES"])
    )
    access_token = create_access_token(
        data={"sub": form_data.username}, expires_delta=access_token_expires
    )
    return Token(access_token=access_token, token_type="bearer")


def sftp_helper(
    storage_control: StorageControl, sftp_source_filename: str, dest_filename: str
):
    storage_control.copy_from_sftp_to_gcs(
        sftp_vars["SFTP_HOST"],
        sftp_vars["SFTP_PORT"],
        sftp_vars["SFTP_USER"],
        sftp_vars["SFTP_PASSWORD"],
        sftp_source_filename,
        get_sftp_bucket_name(env_vars["ENV"]),
        dest_filename,
    )


@app.post("/execute-pdp-pull", response_model=PdpPullResponse)
def execute_pdp_pull(
    req: PdpPullRequest,
    current_username: Annotated[str, Depends(get_current_username)],
    storage_control: Annotated[StorageControl, Depends(StorageControl)],
) -> Any:
    """Performs the PDP pull of the file."""
    storage_control.create_bucket_if_not_exists(get_sftp_bucket_name(env_vars["ENV"]))
    sftp_helper(storage_control, "sftp_file.csv", "write_out_file.csv")
    return {
        "pdp_inst_generated": [],
        "pdp_inst_not_found": [],
    }
