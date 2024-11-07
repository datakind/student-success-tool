"""API functions related to data.
"""

from typing import Annotated, Any, Union
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ..utilities import has_access_to_inst_or_err, has_full_data_access_or_err, BaseUser

router = APIRouter(
    prefix="/institutions",
    tags=["data"],
)

class DataInfo(BaseModel):
    """The Data object that's returned."""
    batch_id: int
    name: str
    record_count: int = 0
    # Size to the nearest MB.
    size: int
    description: str
    # User id of uploader or person who triggered this data ingestion.
    uploader: int
    # Can be PDP_SFTP, MANUAL_UPLOAD etc.
    source: str
    # Disabled data means it is no longer in use.
    data_disabled: bool = False
    # Date in form YYMMDD
    deletion_request: Union[str, None] = None

# Data related operations.

@router.get("/{inst_id}/input_train", response_model=list[DataInfo])
def read_inst_training_inputs(inst_id: int, current_user: Annotated[BaseUser, Depends()]) -> Any:
    """Returns top-level overview of training input data (date uploaded, size, file names etc.).
    
    Only visible to data owners of that institution or higher.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    has_full_data_access_or_err(current_user, "input data")
    return []

@router.get("/{inst_id}/input_train/{batch_id}", response_model=DataInfo)
def read_inst_training_input(
    inst_id: int,
    batch_id: int,
    current_user: Annotated[BaseUser, Depends()]
) -> Any:
    """Returns training input data batch information/details (record count, date uploaded etc.)
    
    Only visible to users of that institution or Datakinder access types.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    has_full_data_access_or_err(current_user, "input data")
    return {
        "batch_id" :batch_id,
        "name": "foo-data", 
        "record_count": 100, 
        "size": 1,
        "description": "some model for foo", 
        "uploader": 123,
        "source": "MANUAL_UPLOAD",
        "data_disabled": False, 
        "deletion_request": None 
    }

@router.get("/{inst_id}/input_exec", response_model=list[DataInfo])
def read_inst_exec_inputs(inst_id: int, current_user: Annotated[BaseUser, Depends()]) -> Any:
    """Returns top-level info on all execution input data (date uploaded, size, file names etc.).
    
    Only visible to users of that institution or Datakinder access types.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    has_full_data_access_or_err(current_user, "input data")
    return []

@router.get("/{inst_id}/input_exec/{batch_id}", response_model=DataInfo)
def read_inst_exec_input(
    inst_id: int,
    batch_id: int,
    current_user: Annotated[BaseUser, Depends()]
) -> Any:
    """Returns a specific batch of execution input data details (record count, date uploaded etc.)
    
    Only visible to users of that institution or Datakinder access types.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    has_full_data_access_or_err(current_user, "input data")
    return {
        "batch_id" :batch_id,
        "name": "foo-data", 
        "record_count": 100, 
        "size": 1,
        "description": "some model for foo", 
        "uploader": 123,
        "source": "MANUAL_UPLOAD",
        "data_disabled": False, 
        "deletion_request": None 
    }
