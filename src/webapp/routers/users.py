"""API functions related to users.
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

router = APIRouter(
    prefix="/api/v1/institutions",
    tags=["users"],
)


class UserAccountRequest(BaseModel):
    """The user account creation request object."""

    # The name can be set by the user
    name: Union[str, None] = None
    access_type: AccessType
    # The email value must be unique across all accounts and provided.
    email: str
    # The username can be set by the user
    username: Union[str, None] = None
    account_disabled: bool = False


class UserAccount(BaseModel):
    """The user account object that's returned."""

    # The user_id will be guaranteed to be unique across all accounts.
    user_id: int
    name: Union[str, None] = None
    inst_id: int
    access_type: AccessType
    # The email value must be unique across all accounts.
    email: str
    # The username value must be unique across all accounts.
    username: Union[str, None] = None
    account_disabled: bool = False
    # Date in form YYMMDD
    deletion_request: Union[str, None] = None


# User account related operations.


@router.get("/{inst_id}/users", response_model=list[UserAccount])
def read_inst_users(inst_id: int, current_user: Annotated[BaseUser, Depends()]) -> Any:
    """Returns all users attributed to a given institution and account type.

    Only visible to data owners of that institution or higher.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    has_full_data_access_or_err(current_user, "users")
    return []


# TODO: Create a way to bulk create users?


@router.post("/{inst_id}/users/", response_model=UserAccount)
def create_new_user(
    inst_id: int,
    user_account_request: UserAccountRequest,
    current_user: Annotated[BaseUser, Depends()],
) -> Any:
    """Create a new user for a given institution.

    Note that for Datakinders creating other Datakinder accounts, use
    institution id = 0.

    Args:
        inst_id: the institution id
        user_account_request: the user account creation requested.
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    if not current_user.has_stronger_permissions_than(user_account_request.access_type):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authorized to create a more powerful user.",
        )
    if current_user.is_viewer():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authorized to create another user.",
        )
    # TODO: check if the email exists in the user table, otherwise, create it.
    # Generate a UUID in the user table.
    uuid = 1
    return {
        "user_id": uuid,
        "name": user_account_request.name,
        "inst_id": inst_id,
        "access_type": user_account_request.access_type,
        "email": user_account_request.email,
        "username": user_account_request.username,
    }


@router.get("/{inst_id}/users/{user_id}", response_model=UserAccount)
def read_inst_user(
    inst_id: int, user_id: int, current_user: Annotated[BaseUser, Depends()]
) -> Any:
    """Returns info on a specific user.

    Only visible to data owners of that institution or higher or that specific user.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    if not current_user.has_full_data_access() and current_user.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authorized to view this user.",
        )
    return {
        "user_id": user_id,
        "name": "",
        "inst_id": inst_id,
        "access_type": 1,
        "email": "",
        "username": "",
        "account_disabled": False,
        "deletion_request": None,
    }
