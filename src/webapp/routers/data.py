"""API functions related to data.
"""

from typing import Annotated, Any, Tuple
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import and_, update
import uuid
from datetime import datetime, date
from sqlalchemy.orm import Session
from sqlalchemy.future import select

from ..utilities import (
    has_access_to_inst_or_err,
    has_full_data_access_or_err,
    BaseUser,
    model_owner_and_higher_or_err,
    prepend_env_prefix,
    uuid_to_str,
    str_to_uuid,
    get_current_active_user,
    DataSource,
)

from ..database import get_session, local_session, BatchTable, FileTable

from ..gcsutil import StorageControl

router = APIRouter(
    prefix="/institutions",
    tags=["data"],
)

# the following fields are not allowed to be changed by any user or caller. They are programmatically set.
IMMUTABLE_BATCH_FIELDS = [
    "batch_id",
    "inst_id",
    "creator",
    "created_date",
    "deletion_request_time",
]


class BatchCreationRequest(BaseModel):
    """The Batch creation request."""

    # Must be unique within an institution to avoid confusion
    name: str
    description: str | None = None
    # Disabled data means it is no longer in use or not available for use.
    batch_disabled: bool = False


class BatchInfo(BaseModel):
    """The Batch Data object that's returned."""

    # In order to allow PATCH commands, each field must be marked as nullable.
    batch_id: str | None = None
    inst_id: str | None = None
    file_ids: set[str] = {}
    # Must be unique within an institution to avoid confusion
    name: str | None = None
    description: str | None = None
    # User id of uploader or person who triggered this data ingestion.
    creator: str | None = None
    # Deleted data means this batch has a pending deletion request and can no longer be used.
    deleted: bool | None = None
    # Completed batches means this batch is ready for use. Completed batches will
    # trigger notifications to Datakind.
    # Can be modified after completion, but this information will not re-trigger
    # notifications to Datakind.
    completed: bool | None = None
    # Date in form YYMMDD. Deletion of a batch will apply to all files in a batch,
    # unless the file is present in other batches.
    deletion_request_time: str | None = None
    created_date: datetime | None = None


class DataInfo(BaseModel):
    """The Data object that's returned. Generally maps to a file, but technically maps to a GCS blob."""

    # Must be unique within an institution to avoid confusion.
    name: str
    data_id: str
    # The batch(es) that this data is present in.
    batch_ids: set[str] = {}
    inst_id: str
    # Size to the nearest MB.
    # size_mb: int
    description: str | None = None
    # User id of uploader or person who triggered this data ingestion. For SST generated files, this field would be null.
    uploader: str | None = None
    # Can be PDP_SFTP, MANUAL_UPLOAD etc.
    source: DataSource | None = None
    # Deleted data means this file has a pending deletion request or is deleted and can no longer be used.
    deleted: bool = False
    # Date in form YYMMDD
    deletion_request_time: date | None = None
    # How long to retain the data.
    # By default (None) -- it is deleted after a successful run. For training dataset it
    # is deleted after the trained model is approved. For inference input, it is deleted
    # after the inference run occurs. For inference output, it is retained indefinitely
    # unless an ad hoc deletion request is received. The type of data is determined by
    # the storage location.
    retention_days: int | None = None
    # Whether the file was generated by SST. (e.g. was it input or output)
    sst_generated: bool
    # Whether the file was validated (in the case of input) or approved (in the case of output).
    valid: bool = False
    uploaded_date: datetime


class DataOverview(BaseModel):
    batches: list[BatchInfo]
    files: list[DataInfo]


# Data related operations. Input files mean files sourced from the institution. Output files are generated by SST.


def get_all_files(
    inst_id: str,
    datakind_user: bool,
    sst_generated_value: bool | None,
    sess: Session,
) -> list[DataInfo]:
    # construct query
    query = None
    if sst_generated_value is None:
        if datakind_user:
            query = select(FileTable).where(
                FileTable.inst_id == str_to_uuid(inst_id),
            )
        else:
            query = select(FileTable).where(
                and_(
                    FileTable.valid,
                    FileTable.inst_id == str_to_uuid(inst_id),
                )
            )
    else:
        if datakind_user:
            query = select(FileTable).where(
                and_(
                    FileTable.inst_id == str_to_uuid(inst_id),
                    FileTable.sst_generated == sst_generated_value,
                )
            )
        else:
            query = select(FileTable).where(
                and_(
                    FileTable.valid,
                    FileTable.inst_id == str_to_uuid(inst_id),
                    FileTable.sst_generated == sst_generated_value,
                )
            )

    result_files = []
    for e in sess.execute(query).all():
        elem = e[0]
        result_files.append(
            {
                "name": elem.name,
                "data_id": uuid_to_str(elem.id),
                "batch_ids": uuids_to_strs(elem.batches),
                "inst_id": uuid_to_str(elem.inst_id),
                # "size_mb": elem.size_mb,
                "description": elem.description,
                "uploader": uuid_to_str(elem.uploader),
                "source": elem.source,
                "deleted": False if elem.deleted is None else elem.deleted,
                "deletion_request_time": elem.deleted_at,
                "retention_days": elem.retention_days,
                "sst_generated": elem.sst_generated,
                "valid": elem.valid,
                "uploaded_date": elem.created_at,
            }
        )
    return result_files


# Some batches are associated with output. This function lets you decide if you want only those batches.
def get_all_batches(
    inst_id: str, datakind_user: bool, output_batches_only: bool, sess: Session
) -> list[BatchInfo]:
    query_result_batches = (
        local_session.get()
        .execute(select(BatchTable).where(BatchTable.inst_id == str_to_uuid(inst_id)))
        .all()
    )
    result_batches = []
    for e in query_result_batches:
        # Note that batches may show file ids of invalid or unapproved files.
        # And will show input and output files.
        # TODO: is this the behavior we want?
        elem = e[0]
        if output_batches_only:
            output_files = [x for x in elem.files if x.sst_generated]
            if not output_files:
                continue
        result_batches.append(
            {
                "batch_id": uuid_to_str(elem.id),
                "inst_id": uuid_to_str(elem.inst_id),
                "name": elem.name,
                "description": elem.description,
                "file_ids": uuids_to_strs(elem.files),
                "creator": uuid_to_str(elem.creator),
                "deleted": False if elem.deleted is None else elem.deleted,
                "completed": False if elem.completed is None else elem.completed,
                "deletion_request_time": elem.deleted_at,
                "created_date": elem.created_at,
            }
        )
    return result_batches


# the input is of type sqlalchemy.orm.collections.InstrumentedSet
def uuids_to_strs(files) -> set[str]:
    return [uuid_to_str(x.id) for x in files]


def strs_to_uuids(files) -> set[uuid.UUID]:
    return [str_to_uuid(x) for x in files]


@router.get("/{inst_id}/input", response_model=DataOverview)
def read_inst_all_input_files(
    inst_id: str,
    current_user: Annotated[BaseUser, Depends(get_current_active_user)],
    sql_session: Annotated[Session, Depends(get_session)],
) -> Any:
    """Returns top-level overview of input data (date uploaded, size, file names etc.).

    Only visible to data owners of that institution or higher.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    has_full_data_access_or_err(current_user, "input data")
    # Datakinders can see unapproved files as well.
    local_session.set(sql_session)
    return {
        "batches": get_all_batches(
            inst_id, current_user.is_datakinder, False, local_session.get()
        ),
        # Set sst_generated_value=false to get input only
        "files": get_all_files(
            inst_id, current_user.is_datakinder, False, local_session.get()
        ),
    }


@router.get("/{inst_id}/output", response_model=DataOverview)
def read_inst_all_output_files(
    inst_id: str,
    current_user: Annotated[BaseUser, Depends(get_current_active_user)],
    sql_session: Annotated[Session, Depends(get_session)],
) -> Any:
    """Returns top-level overview of input data (date uploaded, size, file names etc.) and batch info.

    Only visible to data owners of that institution or higher.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    has_full_data_access_or_err(current_user, "output data")
    local_session.set(sql_session)
    return {
        # Set output_batches_only=true to get output related batches only.
        "batches": get_all_batches(
            inst_id, current_user.is_datakinder, True, local_session.get()
        ),
        # Set sst_generated_value=true to get output only.
        "files": get_all_files(
            inst_id, current_user.is_datakinder, True, local_session.get()
        ),
    }


@router.get("/{inst_id}/batch/{batch_id}", response_model=DataOverview)
def read_batch_info(
    inst_id: str,
    batch_id: str,
    current_user: Annotated[BaseUser, Depends(get_current_active_user)],
    sql_session: Annotated[Session, Depends(get_session)],
) -> Any:
    """Returns batch info and files in that batch.

    Only visible to users of that institution or Datakinder access types.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    has_full_data_access_or_err(current_user, "batch data")
    local_session.set(sql_session)
    query_result = (
        local_session.get()
        .execute(
            select(BatchTable).where(
                and_(
                    BatchTable.id == str_to_uuid(batch_id),
                    BatchTable.inst_id == str_to_uuid(inst_id),
                )
            )
        )
        .all()
    )
    if not query_result or len(query_result) == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No such batch exists.",
        )
    if len(query_result) > 1:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch duplicates found.",
        )
    res = query_result[0][0]
    batch_info = {
        "batch_id": uuid_to_str(res.id),
        "inst_id": uuid_to_str(res.inst_id),
        "name": res.name,
        "description": res.description,
        "file_ids": uuids_to_strs(res.files),
        "creator": uuid_to_str(res.creator),
        "deleted": False if res.deleted is None else res.deleted,
        "completed": False if res.completed is None else res.completed,
        "deletion_request_time": res.deleted_at,
        "created_date": res.created_at,
    }
    data_infos = []
    for elem in res.files:
        data_infos.append(
            {
                "name": elem.name,
                "data_id": uuid_to_str(elem.id),
                "batch_ids": uuids_to_strs(elem.batches),
                "inst_id": uuid_to_str(elem.inst_id),
                # "size_mb": elem.size_mb,
                "description": elem.description,
                "uploader": uuid_to_str(elem.uploader),
                "source": elem.source,
                "deleted": False if elem.deleted is None else elem.deleted,
                "deletion_request_time": elem.deleted_at,
                "retention_days": elem.retention_days,
                "sst_generated": elem.sst_generated,
                "valid": elem.valid,
                "uploaded_date": elem.created_at,
            }
        )
    return {"batches": [batch_info], "files": data_infos}


@router.post("/{inst_id}/batch", response_model=BatchInfo)
def create_batch(
    inst_id: str,
    req: BatchCreationRequest,
    current_user: Annotated[BaseUser, Depends(get_current_active_user)],
    sql_session: Annotated[Session, Depends(get_session)],
) -> Any:
    """Create a new batch.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    model_owner_and_higher_or_err(current_user, "batch")
    local_session.set(sql_session)
    query_result = (
        local_session.get()
        .execute(select(BatchTable).where(BatchTable.name == req.name))
        .all()
    )
    if len(query_result) == 0:
        local_session.get().add(
            BatchTable(
                name=req.name,
                inst_id=str_to_uuid(inst_id),
                description=req.description,
                creator=str_to_uuid(current_user.user_id),
            )
        )
        query_result = (
            local_session.get()
            .execute(select(BatchTable).where(BatchTable.name == req.name))
            .all()
        )
        if not query_result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database write of the batch creation failed.",
            )
        elif len(query_result) > 1:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database write of the batch created duplicate entries.",
            )
    if len(query_result) > 1:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch duplicates found.",
        )
    return {
        "batch_id": uuid_to_str(query_result[0][0].id),
        "inst_id": uuid_to_str(query_result[0][0].inst_id),
        "name": query_result[0][0].name,
        "description": query_result[0][0].description,
        "file_ids": [],
        "creator": uuid_to_str(query_result[0][0].creator),
        "deleted": False,
        "completed": False,
        "deletion_request_time": None,
        "created_date": query_result[0][0].created_at,
    }


def construct_modify_query(modify_vals: dict, batch_id: str) -> Any:
    query = update(BatchTable).where(BatchTable.id == str_to_uuid(batch_id))
    if "name" in modify_vals:
        query.values(name=modify_vals["name"])
    if "description" in modify_vals:
        query.values(description=modify_vals["description"])
    if "deleted" in modify_vals:
        if modify_vals["deleted"]:
            query.values(deleted_at=func.now())
        query.values(deleted=modify_vals["deleted"])
    if "completed" in modify_vals:
        query.values(completed=modify_vals["completed"])
    return query


@router.patch("/{inst_id}/batch/{batch_id}", response_model=BatchInfo)
def update_batch(
    inst_id: str,
    batch_id: str,
    request: BatchInfo,
    current_user: Annotated[BaseUser, Depends(get_current_active_user)],
    sql_session: Annotated[Session, Depends(get_session)],
) -> Any:
    """Modifies an existing batch. Only some fields are allowed to be modified.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    model_owner_and_higher_or_err(current_user, "modify batch")

    update_data = request.model_dump(exclude_unset=True)
    if [key for key in IMMUTABLE_BATCH_FIELDS if key in update_data] != []:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Immutable fields included in modify request.",
        )
    local_session.set(sql_session)
    # Check that the batch exists.
    query_result = (
        local_session.get()
        .execute(select(BatchTable).where(BatchTable.id == str_to_uuid(batch_id)))
        .all()
    )
    if not query_result or len(query_result) == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Batch not found.",
        )
    elif len(query_result) > 1:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Multiple batches with same unique id found.",
        )
    existing_batch = query_result[0][0]
    if existing_batch.deleted:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Batch is set for deletion, no modifications allowed.",
        )
    if "file_ids" in update_data:
        existing_batch.files.clear()
        for f in strs_to_uuids(update_data["file_ids"]):
            # Check that the files requested for this batch exists
            query_result_file = (
                local_session.get()
                .execute(select(FileTable).where(FileTable.id == f))
                .all()
            )
            if not query_result_file or len(query_result_file) == 0:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="file in request not found.",
                )
            elif len(query_result_file) > 1:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Multiple files in request with same unique id found.",
                )
            existing_batch.files.add(query_result_file[0][0])
    # The below is unfortunate but it doesn't seem to work if we programmatically construct the query using values.
    if "name" in update_data:
        local_session.get().execute(
            update(BatchTable)
            .where(BatchTable.id == str_to_uuid(batch_id))
            .values(name=update_data["name"])
        )
    if "description" in update_data:
        local_session.get().execute(
            update(BatchTable)
            .where(BatchTable.id == str_to_uuid(batch_id))
            .values(description=update_data["description"])
        )
    if "deleted" in update_data and update_data["deleted"]:
        # if the user tries to set deleted to false, that is a noop. Deletions can't be undone.
        local_session.get().execute(
            update(BatchTable)
            .where(BatchTable.id == str_to_uuid(batch_id))
            .values(deleted=update_data["deleted"])
            .values(deleted_at=func.now())
        )
    if "completed" in update_data:
        local_session.get().execute(
            update(BatchTable)
            .where(BatchTable.id == str_to_uuid(batch_id))
            .values(completed=update_data["completed"])
        )
    res = (
        local_session.get()
        .execute(select(BatchTable).where(BatchTable.id == str_to_uuid(batch_id)))
        .all()
    )
    return {
        "batch_id": uuid_to_str(res[0][0].id),
        "inst_id": uuid_to_str(res[0][0].inst_id),
        "name": res[0][0].name,
        "description": res[0][0].description,
        "file_ids": uuids_to_strs(res[0][0].files),
        "creator": uuid_to_str(res[0][0].creator),
        "deleted": res[0][0].deleted,
        "completed": res[0][0].completed,
        "deletion_request_time": res[0][0].deleted_at,
        "created_date": res[0][0].created_at,
    }


@router.get("/{inst_id}/file/{file_id}", response_model=DataInfo)
def read_file_info(
    inst_id: str,
    file_id: str,
    current_user: Annotated[BaseUser, Depends(get_current_active_user)],
    sql_session: Annotated[Session, Depends(get_session)],
) -> Any:
    """Returns details on a given file.

    Only visible to users of that institution or Datakinder access types.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    has_full_data_access_or_err(current_user, "file data")
    local_session.set(sql_session)
    query_result = (
        local_session.get()
        .execute(
            select(FileTable).where(
                and_(
                    FileTable.id == str_to_uuid(file_id),
                    FileTable.inst_id == str_to_uuid(inst_id),
                )
            )
        )
        .all()
    )
    # This should only result in a match of a single file.
    if not query_result or len(query_result) == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found.",
        )
    if len(query_result) > 1:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="File duplicates found.",
        )
    res = query_result[0][0]
    return {
        "name": res.name,
        "data_id": uuid_to_str(res.id),
        "batch_ids": uuids_to_strs(res.batches),
        "inst_id": uuid_to_str(res.inst_id),
        # "size_mb": res.size_mb,
        "description": res.description,
        "uploader": uuid_to_str(res.uploader),
        "source": res.source,
        "deleted": False if res.deleted is None else res.deleted,
        "deletion_request_time": res.deleted_at,
        "retention_days": res.retention_days,
        "sst_generated": res.sst_generated,
        "valid": res.valid,
        "uploaded_date": res.created_at,
    }


# TODO: ADD TESTS for the below and finish implementing the below
@router.get("/{inst_id}/file/{file_id}/download", response_model=DataInfo)
def download_inst_file(
    inst_id: str,
    file_id: str,
    current_user: Annotated[BaseUser, Depends(get_current_active_user)],
    sql_session: Annotated[Session, Depends(get_session)],
) -> Any:
    """Enables download of approved output files.

    Only visible to users of that institution or Datakinder access types.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    has_full_data_access_or_err(current_user, "file data")
    local_session.set(sql_session)
    query_result = (
        local_session.get()
        .execute(
            select(FileTable).where(
                and_(
                    FileTable.id == str_to_uuid(file_id),
                    FileTable.inst_id == str_to_uuid(inst_id),
                )
            )
        )
        .all()
    )
    # This should only result in a match of a single file.
    if not query_result or len(query_result) == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found.",
        )
    if len(query_result) > 1:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="File duplicates found.",
        )
    res = query_result[0][0]
    if res.deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File has been deleted.",
        )
    if not res.sst_generated:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Only SST generated files can be downloaded.",
        )
    if res.valid or current_user.is_datakinder:
        # download the file
        bucket_name = prepend_env_prefix(str(res.inst_id))
        file_name = "output/approved/" + res.name
        dest = (
            "Downloads/" + res.name
        )  # xxx TODO update?? Do we want to use signed url for downloads?
        try:
            download_file(bucket_name, file_name, dest)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found:" + str(e),
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="File not yet approved by Datakind. Cannot be downloaded.",
        )
    res = query_result[0][0]
    return {
        "name": res.name,
        "data_id": uuid_to_str(res.id),
        "batch_ids": uuids_to_strs(res.batches),
        "inst_id": uuid_to_str(res.inst_id),
        # "size_mb": res.size_mb,
        "description": res.description,
        "uploader": uuid_to_str(res.uploader),
        "source": res.source,
        "deleted": False if res.deleted is None else res.deleted,
        "deletion_request_time": res.deleted_at,
        "retention_days": res.retention_days,
        "sst_generated": res.sst_generated,
        "valid": res.valid,
        "uploaded_date": res.created_at,
    }


@router.post("/{inst_id}/input/uploadfile", response_model=DataInfo)
def upload_file(
    inst_id: str, current_user: Annotated[BaseUser, Depends(get_current_active_user)]
) -> Any:
    """Add new data from local filesystem.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    model_owner_and_higher_or_err(current_user, "model training data")
    # generate_upload_signed_url(str(inst_id), f"training_data/FOO.csv")
    # TODO: make the POST call to the upload url with the file.
    # Update or create batch.
    return {
        "name": "TEST_UPLOAD_NAME",
    }


@router.post("/{inst_id}/input/pdp_sftp")
def pull_pdp_sftp(
    inst_id: str, current_user: Annotated[BaseUser, Depends(get_current_active_user)]
) -> Any:
    """Add new data from PDP directly.

    This post function triggers a file request to PDP's SFTP server.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    model_owner_and_higher_or_err(current_user, "data")
    # TODO: call function that handles PDP SFTP request here.
