"""API functions related to models.
"""

from typing import Annotated
from fastapi import HTTPException, status, APIRouter

router = APIRouter()

# Model related operations. Or model specific data.

@router.get("/institutions/{inst_id}/models", tags=["models"])
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

@router.get("/institutions/{inst_id}/models/{model_id}", tags=["models"])
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

@router.get("/institutions/{inst_id}/models/{model_id}/vers", tags=["models"])
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

@router.get("/institutions/{inst_id}/models/{model_id}/vers/{vers_id}", tags=["models"])
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

@router.get("/institutions/{inst_id}/models/{model_id}/output", tags=["models"])
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

@router.get("/institutions/{inst_id}/models/{model_id}/output/{output_id}", tags=["models"])
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
