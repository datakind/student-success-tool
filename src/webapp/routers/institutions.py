"""API functions related to institutions.
"""

from typing import Annotated, Any, Union
from fastapi import HTTPException, status, APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy.future import select
import uuid
from google.cloud import storage
from google.cloud.storage import Client

from ..utilities import (
    has_access_to_inst_or_err,
    has_full_data_access_or_err,
    BaseUser,
    AccessType,
    prepend_env_prefix,
    str_to_uuid,
    uuid_to_str,
    get_current_active_user,
)

from ..gcsutil import StorageControl
from ..database import (
    get_session,
    InstTable,
    AccountTable,
    AccountHistoryTable,
    local_session,
)

router = APIRouter(
    tags=["institutions"],
)


# The following are the default top-level folders created in a new GCS bucket.
# softdelete/ is the folder where files in soft-deletion (from user requests or retention time-up) are held prior to deletion.
# Files in softdelete/ should not be visible to even datakinders unless they are in a debugging group -- they can view these files from the gcs console.
DEFAULT_FOLDERS = [
    "input/unvalidated",
    "output/metadata",
    "input/validated",
    "output/unapproved",
    "output/approved",
    "softdelete",  # TODO: we might not need this folder
]


class InstitutionCreationRequest(BaseModel):
    """Institution data creation request.

    The UUID is autogenerated by the database.
    """

    # The name should be unique amongst all other institutions.
    name: str
    description: str | None = None
    retention_days: int | None = None


class Institution(BaseModel):
    """Institution data object."""

    inst_id: str
    name: str
    description: str | None = None
    # The following are characteristics of an institution set at institution creation time.
    # If zero, it follows DK defaults (deletion after completion).
    retention_days: int | None = None  # In Days


@router.get("/institutions", response_model=list[Institution])
def read_all_inst(
    current_user: Annotated[BaseUser, Depends(get_current_active_user)],
    sql_session: Annotated[Session, Depends(get_session)],
) -> Any:
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
    local_session.set(sql_session)
    query_result = local_session.get().execute(select(InstTable))
    res = []
    for elem in query_result:
        res.append(
            {
                "inst_id": uuid_to_str(elem[0].id),
                "name": elem[0].name,
                "description": elem[0].description,
                "retention_days": elem[0].retention_days,
                # TODO add datetime for creation times
            }
        )
    return res


@router.post("/institutions", response_model=Institution)
def create_institution(
    req: InstitutionCreationRequest,
    current_user: Annotated[BaseUser, Depends(get_current_active_user)],
    sql_session: Annotated[Session, Depends(get_session)],
    storage_control: Annotated[StorageControl, Depends(StorageControl)],
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
    local_session.set(sql_session)
    query_result = (
        local_session.get()
        .execute(select(InstTable).where(InstTable.name == req.name))
        .all()
    )
    if len(query_result) == 0:
        # If the institution does not exist create it and create a storage bucket for it.
        # TODO Check presence of storage bucket, if it does not exist, create it.
        local_session.get().add(
            InstTable(
                name=req.name,
                retention_days=req.retention_days,
                description=req.description,
            )
        )
        query_result = (
            local_session.get()
            .execute(select(InstTable).where(InstTable.name == req.name))
            .all()
        )
        if not query_result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database write of the institution creation failed.",
            )
        elif len(query_result) > 1:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database write of the institution created duplicate entries.",
            )
        # Create a storage bucket for it
        bucket_name = prepend_env_prefix(str(query_result[0][0].id))
        try:
            storage_control.create_bucket(bucket_name)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Storage bucket creation failed, storage already exists.",
            )
        # Create the initial folders:
        storage_control.create_folders(bucket_name, DEFAULT_FOLDERS)
    if len(query_result) > 1:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Institution duplicates found.",
        )
    return {
        "inst_id": uuid_to_str(query_result[0][0].id),
        "name": query_result[0][0].name,
        "description": query_result[0][0].description,
        "retention_days": query_result[0][0].retention_days,
    }


# All other API transactions require the UUID as an identifier, this allows the UUID lookup by human readable name.
@router.get("/institutions/name/{inst_name}", response_model=Institution)
def read_inst_name(
    inst_name: str,
    current_user: Annotated[BaseUser, Depends(get_current_active_user)],
    sql_session: Annotated[Session, Depends(get_session)],
) -> Any:
    """Returns overview data on a specific institution.

    The root-level API view. Only visible to users of that institution or Datakinder access types.

    Args:
        current_user: the user making the request.
    """
    local_session.set(sql_session)
    query_result = (
        local_session.get()
        .execute(select(InstTable).where(InstTable.name == inst_name))
        .all()
    )

    if len(query_result) == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Institution not found.",
        )
    if len(query_result) > 1:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Institution duplicates found.",
        )
    has_access_to_inst_or_err(uuid_to_str(query_result[0][0].id), current_user)
    return {
        "inst_id": uuid_to_str(query_result[0][0].id),
        "name": query_result[0][0].name,
        "description": query_result[0][0].description,
        "retention_days": query_result[0][0].retention_days,
    }


@router.get("/institutions/{inst_id}", response_model=Institution)
def read_inst_id(
    inst_id: str,
    current_user: Annotated[BaseUser, Depends(get_current_active_user)],
    sql_session: Annotated[Session, Depends(get_session)],
) -> Any:
    """Returns overview data on a specific institution.

    The root-level API view. Only visible to users of that institution or Datakinder access types.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    local_session.set(sql_session)
    query_result = (
        local_session.get()
        .execute(select(InstTable).where(InstTable.id == str_to_uuid(inst_id)))
        .all()
    )
    if not query_result or len(query_result) == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Institution not found.",
        )
    if len(query_result) > 1:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Institution duplicates found.",
        )
    return {
        "inst_id": uuid_to_str(query_result[0][0].id),
        "name": query_result[0][0].name,
        "description": query_result[0][0].description,
        "retention_days": query_result[0][0].retention_days,
    }
