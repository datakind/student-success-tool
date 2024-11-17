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
    prefix="/institutions",
    tags=["users"],
)


class UserAccount(BaseModel):
    """The user account object that's returned."""

    user_id: int
    name: str
    inst_id: int
    access_type: AccessType
    email: str
    username: str
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
