"""Main file for the SST API.
"""

from datetime import datetime, timedelta, timezone
from typing import Annotated, Union
from enum import Enum

import jwt
from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel

# Store in a python package usable by both the frontend and API
class AccessType(Enum):
    UNKNOWN = 0
    DATAKINDER = 1
    MODEL_OWNER = 2
    DATA_OWNER = 3
    VIEWER = 4

# BaseUser represents an access type. The frontend will use a more detailed User type.
class BaseUser(BaseModel):
    # user_id is permanent and each frontend orginated account will map to a unique user_id.
    # Bare API callers may or may not include a user_id.
    user_id: Union[int, None] = None
    institution: int
    access_type: AccessType

app = FastAPI(
    servers=[
        # TODO: placeholders
        {"url": "https://stag.example.com", "description": "Staging environment"},
        {"url": "https://prod.example.com", "description": "Production environment"},
    ],
    root_path="/api/v1",
)

@app.get("/institution")
def read_all_institutions(
    current_user: Annotated[BaseUser],
):
    """Returns overview data on all institutions.
    
    Only visible to Datakinders.

    Args:
        current_user: the user making the request.
    """
    if current_user.access_type != DATAKINDER:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authorized to read this resource. Select a specific institution.",
        )
    return ""

@app.get("/institution/{inst_id}")
def read_institution(
    current_user: Annotated[BaseUser],
):
    """Returns overview data on a specific institution.
    
    The root-level API view. Only visible to users of that institution or Datakinder access types.

    Args:
        current_user: the user making the request.
    """
    if current_user.institution != inst_id and current_user.access_type != DATAKINDER:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authorized to read this institution's resources.",
        )
    return ""
