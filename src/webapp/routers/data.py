"""API functions related to data.
"""

from typing import Annotated
from fastapi import HTTPException, status, APIRouter

router = APIRouter()
# Data related operations.

@router.get("/institutions/{inst_id}/input_train", tags=["data"])
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
    
@router.get("/institutions/{inst_id}/input_train/{batch_id}", tags=["data"])
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

@router.get("/institutions/{inst_id}/input_exec", tags=["data"])
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

@router.get("/institutions/{inst_id}/input_exec/{batch_id}", tags=["data"])
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
