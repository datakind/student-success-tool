"""
Helper functions that use storage and session.
"""

import uuid

from typing import Annotated, Any, Tuple, Dict
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import and_, or_, update
from datetime import datetime, date
from sqlalchemy.orm import Session
from sqlalchemy.future import select
from ..validation import validate_file_reader
from sqlalchemy.sql import func

from ..utilities import (
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

from ..database import (
    get_session,
    local_session,
    BatchTable,
    FileTable,
    InstTable,
    JobTable,
)


# From a fully qualified file nam (i.e. everything sub-bucket name level), get the job id.
def get_job_id(filename: str) -> int:
    filename.removeprefix("approved/")
    return filename.split("/")[0]


# Updates the sql tables by checking if there are new files in the bucket.
def update_db_from_bucket(inst_id: str, session, storage_control):
    dir_prefix = "approved/"
    res = storage_control.list_blobs_in_folder(
        get_external_bucket_name(inst_id), dir_prefix
    )
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    print(res)
    new_files_to_add_to_database = []
    for f in res:
        # We only handle png and csv outputs.
        if f.endswith(".png") or f.endswith(".csv"):
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
                        valid=True,
                    )
                )
                if f.endswith(".csv"):
                    query = (
                        update(JobTable)
                        .where(JobTable.id == get_job_id(f))
                        .values(output_filename=f, completed=True)
                    )
                    session.execute(query)
    for elem in new_files_to_add_to_database:
        session.add(elem)
