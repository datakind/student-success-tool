"""Main file for the SST API.
"""

from typing import Annotated, Union, Any
from fastapi import FastAPI, HTTPException, status, Depends
from pydantic import BaseModel
#import logging

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

# The institution object that's returned.
class Institution(BaseModel):
    """Institution data object."""
    inst_id: int
    name: str
    description: Union[str, None] = None
    # The following are characteristics of an institution set at institution creation time.
    retention: int

@app.get("/institutions", response_model=list[Institution])
def read_all_inst(current_user: Annotated[BaseUser, Depends()]) -> Any:
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
    return []

@app.get("/institutions/{inst_id}", response_model=Institution)
def read_inst(inst_id: int, current_user: Annotated[BaseUser, Depends()]) -> Any:
    """Returns overview data on a specific institution.
    
    The root-level API view. Only visible to users of that institution or Datakinder access types.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    return {"inst_id": inst_id, "name": "foo", "description" : "", "retention": 0}
