"""API functions related to models. !!!!! NEEDS TO BE UPDATED. !!!!!
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
    ModelTable,
)

from ..gcsutil import StorageControl

router = APIRouter(
    prefix="/institutions",
    tags=["models"],
)


class ModelCreationRequest(BaseModel):
    name: str
    # The default is version zero.
    vers_id: int = 0
    description: str | None = None
    # valid = False, means the model is not ready for use.
    valid: bool = False
    schema_configs: str


class ModelInfo(BaseModel):
    """The model object that's returned."""

    m_id: str
    name: str
    inst_id: str
    # The default is version zero.
    vers_id: int = 0
    description: str | None = None
    # User id of created_by.
    created_by: str | None = None
    valid: bool = False
    deleted: bool | None = None


class RunInfo(BaseModel):
    """The RunInfo object that's returned."""

    run_id: str
    vers_id: int = 0
    inst_id: str
    m_id: str
    # user id of the person who executed this run.
    created_by: str | None = None


# Model related operations. Or model specific data.


@router.get("/{inst_id}/models", response_model=list[ModelInfo])
def read_inst_models(
    inst_id: str,
    current_user: Annotated[BaseUser, Depends(get_current_active_user)],
    sql_session: Annotated[Session, Depends(get_session)],
) -> Any:
    """Returns top-level view of all models attributed to a given institution. Returns all
    versions of all models.

    Only visible to data owners of that institution or higher.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    has_full_data_access_or_err(current_user, "models")
    local_session.set(sql_session)
    query_result = (
        local_session.get()
        .execute(
            select(ModelTable).where(
                and_(
                    ModelTable.inst_id == str_to_uuid(inst_id),
                )
            )
        )
        .all()
    )
    res = []
    for elem in query_result:
        res.append(
            {
                "m_id": uuid_to_str(elem[0].id),
                "inst_id": uuid_to_str(elem[0].inst_id),
                "name": elem[0].name,
                "description": elem[0].description,
                "created_by": uuid_to_str(elem[0].created_by),
                "deleted": elem[0].deleted,
                "valid": elem[0].valid,
            }
        )
    return res


@router.post("/{inst_id}/models/", response_model=ModelInfo)
def create_model(
    inst_id: str,
    req: ModelCreationRequest,
    current_user: Annotated[BaseUser, Depends(get_current_active_user)],
    sql_session: Annotated[Session, Depends(get_session)],
) -> Any:
    """Create a new model (kicks off training a new model).

    Only visible to model owners of that institution or higher. This function may take a
    list of training data batch ids.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    model_owner_and_higher_or_err(current_user, "model training")
    local_session.set(sql_session)
    query_result = (
        local_session.get()
        .execute(
            select(ModelTable).where(
                and_(
                    ModelTable.name == req.name,
                    ModelTable.inst_id == str_to_uuid(inst_id),
                    ModelTable.version == 0 if not req.vers_id else req.vers_id,
                )
            )
        )
        .all()
    )
    if len(query_result) == 0:
        model = ModelTable(
            name=req.name,
            inst_id=str_to_uuid(inst_id),
            description=req.description,
            created_by=str_to_uuid(current_user.user_id),
            valid=req.valid,
            version=req.vers_id,
            # TODO xxx plumb through schema configscs
            # schema_configs=req.schema_configs,
        )
        local_session.get().add(model)
        local_session.get().commit()
        query_result = (
            local_session.get()
            .execute(
                select(ModelTable).where(
                    and_(
                        ModelTable.name == req.name,
                        ModelTable.inst_id == str_to_uuid(inst_id),
                    )
                )
            )
            .all()
        )
        if not query_result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database write of the model creation failed.",
            )
        elif len(query_result) > 1:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database write of the model created duplicate entries.",
            )
    if len(query_result) > 1:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model with this name and version already exists.",
        )
    return {
        "m_id": uuid_to_str(query_result[0][0].id),
        "inst_id": uuid_to_str(query_result[0][0].inst_id),
        "name": query_result[0][0].name,
        "description": query_result[0][0].description,
        "created_by": uuid_to_str(query_result[0][0].created_by),
        "deleted": query_result[0][0].deleted,
        "valid": query_result[0][0].valid,
    }


@router.get("/{inst_id}/models/{model_name}", response_model=list[ModelInfo])
def read_inst_model(
    inst_id: str,
    model_name: str,
    current_user: Annotated[BaseUser, Depends(get_current_active_user)],
    sql_session: Annotated[Session, Depends(get_session)],
) -> Any:
    """Returns a specific model's details e.g. model card.

    Only visible to data owners of that institution or higher.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    has_full_data_access_or_err(current_user, "this model")
    local_session.set(sql_session)
    query_result = (
        local_session.get()
        .execute(
            select(ModelTable).where(
                and_(
                    ModelTable.name == model_name,
                    ModelTable.inst_id == str_to_uuid(inst_id),
                )
            )
        )
        .all()
    )
    if len(query_result) == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found.",
        )
    # if multiple found, that means we have more than one version.
    res = []
    for elem in query_result:
        res.append(
            {
                "m_id": uuid_to_str(elem[0].id),
                "inst_id": uuid_to_str(elem[0].inst_id),
                "name": elem[0].name,
                "description": elem[0].description,
                "created_by": uuid_to_str(elem[0].created_by),
                "deleted": elem[0].deleted,
                "valid": elem[0].valid,
            }
        )
    return res


@router.get("/{inst_id}/models/{model_name}/vers/{vers_id}", response_model=ModelInfo)
def read_inst_model_version(
    inst_id: str,
    model_name: str,
    vers_id: int,
    current_user: Annotated[BaseUser, Depends(get_current_active_user)],
    sql_session: Annotated[Session, Depends(get_session)],
) -> Any:
    """Returns details around a version of a given model.

    Only visible to data owners of that institution or higher.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    has_full_data_access_or_err(current_user, "this model")
    local_session.set(sql_session)
    query_result = (
        local_session.get()
        .execute(
            select(ModelTable).where(
                and_(
                    ModelTable.name == model_name,
                    ModelTable.inst_id == str_to_uuid(inst_id),
                    ModelTable.version == vers_id,
                )
            )
        )
        .all()
    )
    if len(query_result) == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model of this version not found.",
        )
    if len(query_result) > 1:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Multiple modesl of the same version found, this should not have happened.",
        )
    elem = query_result[0]
    return {
        "m_id": uuid_to_str(elem[0].id),
        "inst_id": uuid_to_str(elem[0].inst_id),
        "name": elem[0].name,
        "description": elem[0].description,
        "created_by": uuid_to_str(elem[0].created_by),
        "deleted": elem[0].deleted,
        "valid": elem[0].valid,
    }


@router.get(
    "/{inst_id}/models/{model_name}/vers/{vers_id}/run", response_model=list[RunInfo]
)
def read_inst_model_outputs(
    inst_id: str,
    model_name: str,
    vers_id: int,
    current_user: Annotated[BaseUser, Depends(get_current_active_user)],
    sql_session: Annotated[Session, Depends(get_session)],
) -> Any:
    """Returns top-level info around all executions of a given model.

    Only visible to users of that institution or Datakinder access types.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    local_session.set(sql_session)
    query_result = (
        local_session.get()
        .execute(
            select(ModelTable).where(
                and_(
                    ModelTable.name == model_name,
                    ModelTable.inst_id == str_to_uuid(inst_id),
                    ModelTable.version == vers_id,
                )
            )
        )
        .all()
    )
    if len(query_result) == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found.",
        )
    if len(query_result) > 1:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Multiple modesl of the same version found, this should not have happened.",
        )
    res = []
    if query_result[0][0].run_ids:
        for elem in query_result[0][0].run_ids:
            res.append(
                {
                    # xxxx
                    "run_id": "placeholder",
                    "inst_id": uuid_to_str(query_result[0][0].inst_id),
                    "m_id": uuid_to_str(query_result[0][0].id),
                    "created_by": "placeholder",
                    "vers_id": vers_id,
                }
            )
    return res


@router.get(
    "/{inst_id}/models/{model_name}/vers/{vers_id}/run/{run_id}",
    response_model=RunInfo,
)
def read_inst_model_output(
    inst_id: str,
    model_name: str,
    vers_id: int,
    run_id: str,
    current_user: Annotated[BaseUser, Depends(get_current_active_user)],
    sql_session: Annotated[Session, Depends(get_session)],
) -> Any:
    """Returns a given executions of a given model.

    Only visible to users of that institution or Datakinder access types.
    If a viewer has record allowlist restrictions applied, only those records are returned.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    local_session.set(sql_session)
    query_result = (
        local_session.get()
        .execute(
            select(ModelTable).where(
                and_(
                    ModelTable.name == model_name,
                    ModelTable.inst_id == str_to_uuid(inst_id),
                    ModelTable.version == vers_id,
                )
            )
        )
        .all()
    )
    if len(query_result) == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found.",
        )
    if len(query_result) > 1:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Multiple models of the same version found, this should not have happened.",
        )
    if query_result[0][0].run_ids:
        # TODO xxx: check the run id is present, then make a query to Databricks
        return {
            # xxxx
            "run_id": "placeholder",
            "inst_id": uuid_to_str(query_result[0][0].inst_id),
            "m_id": uuid_to_str(query_result[0][0].id),
            "created_by": "placeholder",
            "vers_id": vers_id,
        }
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Run not found.",
    )


# TODO: DK to implement
@router.post("/{inst_id}/models/{model_id}/vers/")
def retrain_model(
    inst_id: str,
    model_id: str,
    current_user: Annotated[BaseUser, Depends(get_current_active_user)],
    sql_session: Annotated[Session, Depends(get_session)],
) -> Any:
    """Retrain an existing model (creates a new version of a model).

    Only visible to model owners of that institution or higher. This function takes a
    list of training data batch ids.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    model_owner_and_higher_or_err(current_user, "model training")
    return
