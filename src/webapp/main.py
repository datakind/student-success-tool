"""Main file for the SST API.
"""

from datetime import datetime, timedelta, timezone
from typing import Annotated
from fastapi import FastAPI, HTTPException, status

from .utilities import has_access_to_inst_or_err, BaseUser
from .routers import models, users, data

app = FastAPI(
    servers=[
        # TODO: placeholders
        {"url": "https://stag.example.com", "description": "Staging environment"},
        {"url": "https://prod.example.com", "description": "Production environment"},
    ],
    root_path="/api/v1",
)

app.include_router(models.router)
app.include_router(users.router)
app.include_router(data.router)

# Institution related operations.

@app.get("/institutions")
def read_all_inst(
    current_user: BaseUser,
):
    """Returns overview data on all institutions.
    
    Only visible to Datakinders.

    Args:
        current_user: the user making the request.
    """
    if not current_user.is_datakinder:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authorized to read this resource. Select a specific institution.",
        )
    return {"message": "All instutions"}

@app.get("/institutions/{inst_id}")
def read_inst(
    current_user: BaseUser,
):
    """Returns overview data on a specific institution.
    
    The root-level API view. Only visible to users of that institution or Datakinder access types.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    return {"message": "instution "+str(inst_id)}
