"""
Helper functions that use storage and session.
"""

import uuid

from typing import Annotated, Any, Tuple, Dict
from pydantic import BaseModel
from sqlalchemy import and_, or_, update
from datetime import datetime, date
from sqlalchemy.orm import Session
from sqlalchemy.future import select
from sqlalchemy.sql import func

from .utilities import (
    has_access_to_inst_or_err,
    has_full_data_access_or_err,
    BaseUser,
    model_owner_and_higher_or_err,
    uuid_to_str,
    str_to_uuid,
    get_current_active_user,
    DataSource,
    get_external_bucket_name,
    SchemaType,
)

from .database import (
    get_session,
    local_session,
    BatchTable,
    FileTable,
    InstTable,
    JobTable,
)


# From a fully qualified file nam (i.e. everything sub-bucket name level), get the job id.
def get_job_id(filename: str) -> int:
    tmp = ""
    if filename.startswith("approved/"):
        tmp = filename.removeprefix("approved/")
    elif filename.startswith("unapproved/"):
        tmp = filename.removeprefix("unapproved/")
    else:
        raise ValueError("Unexpected filename structure.")
    return int(tmp.split("/")[0])


def is_file_approved(filename: str) -> bool:
    if filename.startswith("approved/"):
        return True
    elif filename.startswith("unapproved/"):
        return False
    else:
        raise ValueError("Unexpected filename structure.")


# Updates the sql tables by checking if there are new files in the bucket.
"""
Note that while all output files will be added to the file table, potentially with their own approval status, the JobTable will only refer to the csv inference output and indicate validity (approval) value for that file.
This means that for a single run it's possible to have some output files be approved and some be unapproved but that is confusing and we discourage it.
Note that deleted files are handled upon file retrieval, not here.
"""


def update_db_from_bucket(inst_id: str, session, storage_control):
    dir_prefix = ["approved/", "unapproved/"]
    all_files = []
    for d in dir_prefix:
        all_files = all_files + storage_control.list_blobs_in_folder(
            get_external_bucket_name(inst_id), dir_prefix
        )
    new_files_to_add_to_database = []
    for f in all_files:
        # We only handle png and csv outputs.
        if not f.endswith(".png") and not f.endswith(".csv"):
            continue
        file_approved = is_file_approved(f)
        # Check if that file already exists in the table, otherwise add it.
        query_result = session.execute(
            select(FileTable).where(
                and_(
                    FileTable.name == f,
                    FileTable.inst_id == str_to_uuid(inst_id),
                )
            )
        ).all()
        if len(query_result) == 0:
            new_files_to_add_to_database.append(
                FileTable(
                    name=f,
                    inst_id=str_to_uuid(inst_id),
                    sst_generated=True,
                    schemas=[
                        (
                            SchemaType.PNG
                            if f.endswith(".png")
                            else SchemaType.SST_OUTPUT
                        )
                    ],
                    valid=file_approved,
                )
            )
            if f.endswith(".csv"):
                query = (
                    update(JobTable)
                    .where(JobTable.id == get_job_id(f))
                    .values(
                        output_filename=f, completed=True, output_valid=file_approved
                    )
                )
                session.execute(query)
        elif len(query_result) == 1:
            # This file already exists, check if its status has changed.
            if query_result[0][0].valid != file_approved:
                # Technically you could make an approved file unapproved.
                query = (
                    update(FileTable)
                    .where(FileTable.name == f)
                    .values(valid=file_approved)
                )
                session.execute(query)
        else:
            raise ValueError("Attempted creation of file with duplicate name.")
    for elem in new_files_to_add_to_database:
        session.add(elem)
