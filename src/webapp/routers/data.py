"""API functions related to data.
"""

from typing import Annotated, Any, Union
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ..utilities import (
    has_access_to_inst_or_err,
    has_full_data_access_or_err,
    BaseUser,
    model_owner_and_higher_or_err,
)

router = APIRouter(
    prefix="/api/v1/institutions",
    tags=["data"],
)


class BatchCreationRequest(BaseModel):
    """The Batch Data object that's returned."""

    # The name must be unique (to prevent confusion).
    name: str
    description: Union[str, None] = None
    # Disabled data means it is no longer in use or not available for use.
    batch_disabled: bool = False


class BatchInfo(BaseModel):
    """The Batch Data object that's returned."""

    batch_id: int
    name: str
    description: str
    file_names: list[str]
    # User id of uploader or person who triggered this data ingestion.
    creator: int
    # Disabled data means it is no longer in use.
    batch_disabled: bool = False
    # Date in form YYMMDD
    deletion_request: Union[str, None] = None
    created_date: str


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
def read_inst_training_inputs(
    inst_id: int, current_user: Annotated[BaseUser, Depends()]
) -> Any:
    """Returns top-level overview of training input data (date uploaded, size, file names etc.).

    Only visible to data owners of that institution or higher.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    has_full_data_access_or_err(current_user, "input data")
    return []


@router.post("/{inst_id}/input_train/", response_model=BatchInfo)
def create_batch(
    inst_id: int,
    request: BatchCreationRequest,
    current_user: Annotated[BaseUser, Depends()],
) -> Any:
    """Create a new batch of training data.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    model_owner_and_higher_or_err(current_user, "model training data")
    # TODO: check the batch name does not already exist, if it exists,
    # notify user to update batch instead.
    return {
        "batch_id": 1,
        "name": request.name,
        "description": "",
        "file_names": [],
        "creator": 1,
        "created_date": "mmyydd",
    }


@router.get("/{inst_id}/input_train/{batch_id}", response_model=DataInfo)
def read_inst_training_input(
    inst_id: int, batch_id: int, current_user: Annotated[BaseUser, Depends()]
) -> Any:
    """Returns training input data batch information/details (record count, date uploaded etc.)

    Only visible to users of that institution or Datakinder access types.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    has_full_data_access_or_err(current_user, "input data")
    return {
        "batch_id": batch_id,
        "name": "foo-data",
        "record_count": 100,
        "size": 1,
        "description": "some model for foo",
        "uploader": 123,
        "source": "MANUAL_UPLOAD",
        "data_disabled": False,
        "deletion_request": None,
    }


@router.post("/{inst_id}/input_train/{batch_id}/")
def upload_file(
    inst_id: int, batch_id: int, current_user: Annotated[BaseUser, Depends()]
) -> Any:
    """Add new training data from local filesystem.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    model_owner_and_higher_or_err(current_user, "model training data")
    # generate_upload_signed_url(str(inst_id), f"training_data/FOO.csv")
    # TODO: make the POST call to the upload url with the file.
    # Update or create batch.


@router.post("/{inst_id}/input_train/{batch_id}/pdp_sftp/")
def pull_pdp_sftp(inst_id: int, current_user: Annotated[BaseUser, Depends()]) -> Any:
    """Add new training data from PDP directly.

    This post function triggers a file request to PDP's SFTP server.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    model_owner_and_higher_or_err(current_user, "model training data")
    # TODO: call function that handles PDP SFTP request here.


@router.get("/{inst_id}/input_inference", response_model=list[DataInfo])
def read_inst_inference_inputs(
    inst_id: int, current_user: Annotated[BaseUser, Depends()]
) -> Any:
    """Returns top-level info on all inference input data (date uploaded, size, file names etc.).

    Only visible to users of that institution or Datakinder access types.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    has_full_data_access_or_err(current_user, "inference data")
    return []


@router.post("/{inst_id}/input_inference/", response_model=BatchInfo)
def create_inference_batch(
    inst_id: int,
    request: BatchCreationRequest,
    current_user: Annotated[BaseUser, Depends()],
) -> Any:
    """Create a new batch of inference data.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    has_full_data_access_or_err(current_user, "inference data")
    # TODO: check the batch name does not already exist, if it exists,
    # notify user to update batch instead.
    return {
        "batch_id": 1,
        "name": request.name,
        "description": "",
        "file_names": [],
        "creator": 1,
        "created_date": "mmyydd",
    }


@router.get("/{inst_id}/input_inference/{batch_id}", response_model=DataInfo)
def read_inst_inference_input(
    inst_id: int, batch_id: int, current_user: Annotated[BaseUser, Depends()]
) -> Any:
    """Returns a specific batch of inference input data details (record count, date uploaded etc.)

    Only visible to users of that institution or Datakinder access types.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    has_full_data_access_or_err(current_user, "inference data")
    return {
        "batch_id": batch_id,
        "name": "foo-data",
        "record_count": 100,
        "size": 1,
        "description": "some model for foo",
        "uploader": 123,
        "source": "MANUAL_UPLOAD",
        "data_disabled": False,
        "deletion_request": None,
    }


@router.post("/{inst_id}/input_inference/{batch_id}/")
def upload_inference_file(
    inst_id: int, batch_id: int, current_user: Annotated[BaseUser, Depends()]
) -> Any:
    """Add new inference data from local filesystem.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    has_full_data_access_or_err(current_user, "model inference data")
    # generate_upload_signed_url(str(inst_id), f"inference_data/FOO.bar")
    # TODO: make the POST call to the upload url with the file.
    # Update or create batch.


@router.post("/{inst_id}/input_inference/{batch_id}/pdp_sftp/")
def pull_pdp_sftp_inference(
    inst_id: int, batch_id: int, current_user: Annotated[BaseUser, Depends()]
) -> Any:
    """Add new inference data from PDP directly.

    This post function triggers a file request to PDP's SFTP server.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    has_full_data_access_or_err(current_user, "model inference data")
    # TODO: call function that handles PDP SFTP request here.
