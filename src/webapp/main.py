"""Main file for the SST API.
"""

from datetime import datetime, timedelta, timezone
from typing import Annotated, Union
from enum import Enum

import jwt
from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel

# TODO: Store in a python package to be usable by the frontend.
# Accesstypes in order of decreasing access.
class AccessType(Enum):
    DATAKINDER = 1
    MODEL_OWNER = 2
    DATA_OWNER = 3
    VIEWER = 4

# BaseUser represents an access type. The frontend will include more detailed User info.
class BaseUser(BaseModel):
    # user_id is permanent and each frontend orginated account will map to a unique user_id.
    # Bare API callers will likely not include a user_id.
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

# Private helper functions.

# Whether a given user has access to a given institution.
def has_access_to_inst(inst: int, user: Annotated[BaseUser]) -> bool:
    return user.institution = inst_id or user.access_type != DATAKINDER:

# Whether a given user is a Datakinder.
def is_datakinder(user: Annotated[BaseUser]) -> bool:
    return user.access_type = DATAKINDER

# Datakinders, model_owners, data_owners, all have full data access.
def has_full_data_access(user: Annotated[BaseUser]) -> bool:
    return user.access_type = DATAKINDER or user.access_type = MODEL_OWNER or user.access_type = DATA_OWNER

# Raise error if a given user does not have access to a given institution.
def has_access_to_inst_or_err(inst: int, user: Annotated[BaseUser]):
    if not has_access_to_inst(inst, user):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authorized to read this institution's resources.",
        )
    return

# Raise error if a given user does not have data access to a given institution.
def has_full_data_access_or_err(user: Annotated[BaseUser], resource_type: str):
    if not has_full_data_access(user):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authorized to view " + resource_type + " for this institution.",
        )
    return

# Public API below.

# Institution related operations.

@app.get("/institutions")
def read_all_inst(
    current_user: Annotated[BaseUser],
):
    """Returns overview data on all institutions.
    
    Only visible to Datakinders.

    Args:
        current_user: the user making the request.
    """
    if not is_datakinder(current_user):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authorized to read this resource. Select a specific institution.",
        )
    return ""

@app.get("/institutions/{inst_id}")
def read_inst(
    current_user: Annotated[BaseUser],
):
    """Returns overview data on a specific institution.
    
    The root-level API view. Only visible to users of that institution or Datakinder access types.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    return ""

# Model related operations. Or model specific data.

@app.get("/institutions/{inst_id}/models")
def read_inst_models(
    current_user: Annotated[BaseUser],
):
    """Returns top-level view of all models attributed to a given institution.
    
    Only visible to data owners of that institution or higher.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    has_full_data_access_or_err(current_user, "models")
    return ""

@app.get("/institutions/{inst_id}/models/{model_id}")
def read_inst_model(
    current_user: Annotated[BaseUser],
):
    """Returns a specific model's details e.g. model card.
    
    Only visible to data owners of that institution or higher.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    has_full_data_access_or_err(current_user, "this model")
    return ""

@app.get("/institutions/{inst_id}/models/{model_id}/vers")
def read_inst_model_versions(
    current_user: Annotated[BaseUser],
):
    """Returns all versions of a given model.
    
    Only visible to data owners of that institution or higher. This can include retrained models etc.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    has_full_data_access_or_err(current_user, "this model")
    
    return ""

@app.get("/institutions/{inst_id}/models/{model_id}/vers/{vers_id}")
def read_inst_model_version(
    current_user: Annotated[BaseUser],
):
    """Returns details around a version of a given model.
    
    Only visible to data owners of that institution or higher.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    has_full_data_access_or_err(current_user, "this model")
    return ""

@app.get("/institutions/{inst_id}/models/{model_id}/output")
def read_inst_model_outputs(
    current_user: Annotated[BaseUser],
):
    """Returns top-level info around all executions of a given model.
    
    Only visible to users of that institution or Datakinder access types.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    return ""

@app.get("/institutions/{inst_id}/models/{model_id}/output/{output_id}")
def read_inst_model_output(
    current_user: Annotated[BaseUser],
):
    """Returns a given executions of a given model.
    
    Only visible to users of that institution or Datakinder access types.
    If a viewer has record allowlist restrictions applied, only those records are returned.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    return ""

# Data related operations.

@app.get("/institutions/{inst_id}/input_train")
def read_inst_training_inputs(
    current_user: Annotated[BaseUser],
):
    """Returns top-level overview of training input data (date uploaded, size, file names etc.).
    
    Only visible to data owners of that institution or higher.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    has_full_data_access_or_err(current_user, "input data")
    return ""
    
@app.get("/institutions/{inst_id}/input_train/{batch_id}")
def read_inst_training_input(
    current_user: Annotated[BaseUser],
):
    """Returns training input data batch information/details (record count, date uploaded etc.)
    
    Only visible to users of that institution or Datakinder access types.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    has_full_data_access_or_err(current_user, "input data")
    return ""

@app.get("/institutions/{inst_id}/input_exec")
def read_inst_exec_inputs(
    current_user: Annotated[BaseUser],
):
    """Returns top-level info on all execution input data (date uploaded, size, file names etc.).
    
    Only visible to users of that institution or Datakinder access types.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    has_full_data_access_or_err(current_user, "input data")
    return ""

@app.get("/institutions/{inst_id}/input_exec/{batch_id}")
def read_inst_exec_input(
    current_user: Annotated[BaseUser],
):
    """Returns a specific batch of execution input data details (record count, date uploaded etc.)
    
    Only visible to users of that institution or Datakinder access types.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    has_full_data_access_or_err(current_user, "input data")
    return ""

# User account related operations.
    
@app.get("/institutions/{inst_id}/users")
def read_inst_users(
    current_user: Annotated[BaseUser],
):
    """Returns all users attributed to a given institution and account type.
    
    Only visible to data owners of that institution or higher.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    has_full_data_access_or_err(current_user, "users")
    return ""

@app.get("/institutions/{inst_id}/users/{user_id}")
def read_inst_user(
    current_user: Annotated[BaseUser],
):
    """Returns info on a specific user.
    
    Only visible to data owners of that institution or higher or that specific user.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    if not has_full_data_access(current_user) and current_user.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authorized to view this user.",
        )
    return ""
