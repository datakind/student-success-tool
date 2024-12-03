"""API functions related to institutions.
"""

from typing import Annotated, Any, Union
from fastapi import HTTPException, status, APIRouter, Depends
from pydantic import BaseModel

from ..utilities import (
    has_access_to_inst_or_err,
    has_full_data_access_or_err,
    BaseUser,
    AccessType,
)

from ..upload import generate_upload_signed_url

router = APIRouter(
    tags=["institutions"],
)


class Institution(BaseModel):
    """Institution data object."""

    inst_id: int
    name: str
    description: Union[str, None] = None
    # The following are characteristics of an institution set at institution creation time.
    retention_days: int  # In Days


@router.get("/institutions", response_model=list[Institution])
def read_all_inst(current_user: Annotated[BaseUser, Depends()]) -> Any:
    """Returns overview data on all institutions.

    Only visible to Datakinders.

    Args:
        current_user: the user making the request.
    """
    if not current_user.is_datakinder():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authorized to read this resource. Select a specific institution.",
        )
    return []


@router.post("/institutions/", response_model=Institution)
def create_institution(
    institution: Institution, current_user: Annotated[BaseUser, Depends()]
) -> Any:
    """Create a new institution.

    Only available to Datakinders.

    Args:
        current_user: the user making the request.
    """
    if not current_user.is_datakinder():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authorized to create an institution.",
        )
    # TODO: check if the institution exists in the institution table, otherwise,
    # create it and create a storage bucket for it.
    return institution


@router.get("/institutions/{inst_id}", response_model=Institution)
def read_inst(inst_id: int, current_user: Annotated[BaseUser, Depends()]) -> Any:
    """Returns overview data on a specific institution.

    The root-level API view. Only visible to users of that institution or Datakinder access types.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    return {"inst_id": inst_id, "name": "", "description": "", "retention_days": 0}


@router.get("/institutions/{inst_id}/upload-url", response_model=str)
def get_upload_url(inst_id: str) -> Any:
    """Returns a signed URL for uploading data to a specific institution.

    Args:
        current_user: the user making the request.
    """
    # has_access_to_inst_or_err(inst_id, current_user)
    return generate_upload_signed_url("local-upload-test", f"{inst_id}/test.csv")
