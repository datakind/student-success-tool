"""API functions related to models. !!!!! NEEDS TO BE UPDATED. !!!!!
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
    prefix="/institutions",
    tags=["models"],
)


class Model(BaseModel):
    """The model object that's returned."""

    m_id: int
    name: str
    # The default is version zero.
    vers_id: int = 0
    description: str
    # User id of creator.
    creator: int
    # Disabling a model means it is no longer in use.
    disabled: bool = False
    # Date in form YYMMDD
    deletion_request: Union[str, None] = None


class Execution(BaseModel):
    """The execution object that's returned."""

    m_id: int
    vers_id: int = 0
    output_id: int
    # user id of the person who executed this run.
    executor: int
    # Disabling an execution means it is no longer in use.
    execution_disabled: bool = False
    # Date in form YYMMDD
    deletion_request: Union[str, None] = None


# Model related operations. Or model specific data.


@router.get("/{inst_id}/models", response_model=list[Model])
def read_inst_models(inst_id: int, current_user: Annotated[BaseUser, Depends()]) -> Any:
    """Returns top-level view of all models attributed to a given institution. Returns all
    versions of all models.

    Only visible to data owners of that institution or higher.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    has_full_data_access_or_err(current_user, "models")
    return []


# TODO: Pending timeline, fellows may not get to complete this function.
@router.post("/{inst_id}/models/")
def train_new_model(inst_id: int, current_user: Annotated[BaseUser, Depends()]) -> Any:
    """Create a new model (kicks off training a new model).

    Only visible to model owners of that institution or higher. This function may take a
    list of training data batch ids.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    model_owner_and_higher_or_err(current_user, "model training")


@router.get("/{inst_id}/models/{model_id}", response_model=Model)
def read_inst_model(
    inst_id: int, model_id: int, current_user: Annotated[BaseUser, Depends()]
) -> Any:
    """Returns a specific model's details e.g. model card.

    Only visible to data owners of that institution or higher.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    has_full_data_access_or_err(current_user, "this model")
    # Returns the default model version (0) if not disabled, or returns the first non-disabled
    # version (generally oldest).
    return {
        "m_id": model_id,
        "name": "foo-model",
        "vers_id": 0,
        "description": "some model for foo",
        "creator": 123,
        "disabled": False,
        "deletion_request": None,
    }


@router.get("/{inst_id}/models/{model_id}/vers", response_model=list[Model])
def read_inst_model_versions(
    inst_id: int, model_id: int, current_user: Annotated[BaseUser, Depends()]
) -> Any:
    """Returns all versions of a given model.

    Only visible to data owners of that institution or higher. This can include retrained models
    etc.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    has_full_data_access_or_err(current_user, "this model")
    # Returns all versions of a model (each version is still a model object).
    return []


# TODO: Pending timeline, fellows may not get to complete this function.
@router.post("/{inst_id}/models/{model_id}/vers/")
def retrain_model(
    inst_id: int, model_id: int, current_user: Annotated[BaseUser, Depends()]
) -> Any:
    """Retrain an existing model (creates a new version of a model).

    Only visible to model owners of that institution or higher. This function takes a
    list of training data batch ids.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    model_owner_and_higher_or_err(current_user, "model training")


@router.get("/{inst_id}/models/{model_id}/vers/{vers_id}", response_model=Model)
def read_inst_model_version(
    inst_id: int,
    model_id: int,
    vers_id: int,
    current_user: Annotated[BaseUser, Depends()],
) -> Any:
    """Returns details around a version of a given model.

    Only visible to data owners of that institution or higher.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    has_full_data_access_or_err(current_user, "this model")
    return {
        "m_id": model_id,
        "name": "foo-model",
        "vers_id": vers_id,
        "description": "some model for foo",
        "creator": 123,
        "disabled": False,
        "deletion_request": None,
    }


@router.get(
    "/{inst_id}/models/{model_id}/vers/{vers_id}/output", response_model=list[Execution]
)
def read_inst_model_outputs(
    inst_id: int,
    model_id: int,
    vers_id: int,
    current_user: Annotated[BaseUser, Depends()],
) -> Any:
    """Returns top-level info around all executions of a given model.

    Only visible to users of that institution or Datakinder access types.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    return []


@router.get(
    "/{inst_id}/models/{model_id}/vers/{vers_id}/output/{output_id}",
    response_model=Execution,
)
def read_inst_model_output(
    inst_id: int,
    model_id: int,
    vers_id: int,
    output_id: int,
    current_user: Annotated[BaseUser, Depends()],
) -> Any:
    """Returns a given executions of a given model.

    Only visible to users of that institution or Datakinder access types.
    If a viewer has record allowlist restrictions applied, only those records are returned.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    return {
        "m_id": model_id,
        "vers_id": vers_id,
        "output_id": output_id,
        "executor": 123,
        "execution_disabled": False,
        "deletion_request": None,
    }
