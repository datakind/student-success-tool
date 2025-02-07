"""API functions related to users.
"""

from typing import Annotated, Any, Dict
from fastapi import HTTPException, status, APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy.future import select
from sqlalchemy import and_

from ..utilities import (
    has_access_to_inst_or_err,
    has_full_data_access_or_err,
    BaseUser,
    AccessType,
    get_current_active_user,
    str_to_uuid,
    uuid_to_str,
)

from ..database import get_session, local_session, AccountTable, InstTable

router = APIRouter(
    prefix="/institutions",
    tags=["users"],
)

# TODO: update the user creation flow to check allowed_emails first.


class UserAccountRequest(BaseModel):
    """The user account creation request object."""

    # The name can be set by the user
    name: str | None = None
    access_type: AccessType | None = None
    # The email value must be unique across all accounts and provided.
    email: str


class UserAccount(BaseModel):
    """The user account object that's returned."""

    # The user_id will be guaranteed to be unique across all accounts.
    user_id: str
    name: str | None = None
    inst_id: str
    access_type: AccessType | None = None
    # The email value must be unique across all accounts.
    email: str


# User account related operations.


@router.get("/{inst_id}/users", response_model=list[UserAccount])
def read_inst_users(
    inst_id: str,
    current_user: Annotated[BaseUser, Depends(get_current_active_user)],
    sql_session: Annotated[Session, Depends(get_session)],
) -> Any:
    """Returns all users attributed to a given institution.

    Only visible to data owners of that institution or higher.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    has_full_data_access_or_err(current_user, "users")
    local_session.set(sql_session)
    query_result = (
        local_session.get()
        .execute(
            select(AccountTable).where(
                AccountTable.inst_id == str_to_uuid(inst_id),
            )
        )
        .all()
    )
    if not query_result or len(query_result) == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="user in request not found.",
        )
    result = []
    for e in query_result:
        elem = e[0]
        result.append(
            {
                "user_id": uuid_to_str(elem.id),
                "inst_id": uuid_to_str(elem.inst_id),
                "name": elem.name,
                "access_type": elem.access_type,
                "email": elem.email,
            }
        )
    return result


@router.get("/{inst_id}/users/{user_id}", response_model=UserAccount)
def read_inst_user(
    inst_id: str,
    user_id: str,
    current_user: Annotated[BaseUser, Depends(get_current_active_user)],
    sql_session: Annotated[Session, Depends(get_session)],
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
    local_session.set(sql_session)
    query_result = (
        local_session.get()
        .execute(
            select(AccountTable).where(
                and_(
                    AccountTable.inst_id == str_to_uuid(inst_id),
                    AccountTable.id == str_to_uuid(user_id),
                )
            )
        )
        .all()
    )
    if not query_result or len(query_result) == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="user in request not found.",
        )
    elif len(query_result) > 1:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Multiple users in request with same unique id found.",
        )
    elem = query_result[0][0]
    return {
        "user_id": uuid_to_str(elem.id),
        "inst_id": uuid_to_str(elem.inst_id),
        "name": elem.name,
        "access_type": elem.access_type,
        "email": elem.email,
    }


@router.get("/{inst_id}/allowable-emails", response_model=Dict[str, str])
def read_inst_allowed_emails(
    inst_id: str,
    current_user: Annotated[BaseUser, Depends(get_current_active_user)],
    sql_session: Annotated[Session, Depends(get_session)],
) -> Any:
    """Returns the allowed emails of a given isntitution. These are the emails that can sign up for a given institution.

    Only visible to data owners of that institution or higher or that specific user.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    if not current_user.has_full_data_access() and current_user.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authorized to view user info.",
        )
    local_session.set(sql_session)
    query_result = (
        local_session.get()
        .execute(
            select(InstTable).where(
                InstTable.id == str_to_uuid(inst_id),
            )
        )
        .all()
    )
    if not query_result or len(query_result) == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Institution in request not found.",
        )
    elif len(query_result) > 1:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Institution duplicates found.",
        )
    elem = query_result[0][0].allowed_emails
    if not elem:
        return {}
    return elem


# TODO: xxx finish and also test
@router.patch("/{inst_id}/user/{user_id}", response_model=UserAccount)
def update_inst_user(
    inst_id: str,
    user_id: str,
    user_account_request: UserAccount,
    current_user: Annotated[BaseUser, Depends(get_current_active_user)],
    sql_session: Annotated[Session, Depends(get_session)],
) -> Any:
    """Returns the allowed emails of a given isntitution. These are the emails that can sign up for a given institution.

    Only visible to data owners of that institution or higher or that specific user.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    if not current_user.has_full_data_access() and current_user.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authorized to edit user info.",
        )
    local_session.set(sql_session)
    query_result = (
        local_session.get()
        .execute(
            select(AccountTable).where(
                and_(
                    AccountTable.inst_id == str_to_uuid(inst_id),
                    AccountTable.id == str_to_uuid(user_id),
                )
            )
        )
        .all()
    )
    if not query_result or len(query_result) == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Institution in request not found.",
        )
    elif len(query_result) > 1:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Institution duplicates found.",
        )
    elem = query_result[0][0]
    # TODO: patch
    return {
        "user_id": uuid_to_str(elem.id),
        "inst_id": uuid_to_str(elem.inst_id),
        "name": elem.name,
        "access_type": elem.access_type,
        "email": elem.email,
    }


# TODO: Create a way to bulk create users?


# TODO delete? or make this only for backend cases
# Currently has no test cases
@router.post("/{inst_id}/users", response_model=UserAccount)
def create_new_users(
    inst_id: str,
    user_account_request: list[UserAccountRequest],
    current_user: Annotated[BaseUser, Depends(get_current_active_user)],
    sql_session: Annotated[Session, Depends(get_session)],
) -> Any:
    """Create a list of new users for a given institution.

    Note that for Datakinders creating other Datakinder accounts, use separate endpoint /datakinders (see main.py). This is for NON datakinder accounts.

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
    return {
        "user_id": "",
        "name": user_account_request.name,
        "inst_id": inst_id,
        "access_type": user_account_request.access_type,
        "email": user_account_request.email,
    }
