"""Helper functions that may be used across multiple API router subpackages.
"""

from typing import Annotated
# the following needed for python pre 3.11
from strenum import StrEnum
import uuid
import os
import jwt

from fastapi import HTTPException, status, Depends
from pydantic import BaseModel, ConfigDict
from jwt.exceptions import InvalidTokenError
from sqlalchemy.orm import Session
from datetime import datetime, timedelta, timezone
from sqlalchemy.future import select

from .authn import verify_password, TokenData, oauth2_scheme
from .database import get_session, AccountTable
from .config import env_vars


# TODO: Store in a python package to be usable by the frontend.
class AccessType(StrEnum):
    """Access types available."""

    DATAKINDER = "DATAKINDER"
    MODEL_OWNER = "MODEL_OWNER"
    DATA_OWNER = "DATA_OWNER"
    VIEWER = "VIEWER"


class DataSource(StrEnum):
    """Where the Data was created from."""

    UNNOWN = "UNKNOWN"
    PDP_SFTP = "PDP_SFTP"
    MANUAL_UPLOAD = "MANUAL_UPLOAD"


class BaseUser(BaseModel):
    """BaseUser represents an access type. The frontend will include more detailed User info."""

    model_config = ConfigDict(use_enum_values=True)
    # user_id is permanent and each frontend orginated account will map to a unique user_id.
    # Bare API callers will likely not include a user_id.
    # The actual types of the ids will be UUIDs.
    user_id: str | None = None
    email: str | None = None
    # For Datakinders, institution is None which means "no inst specified".
    institution: str | None = None
    access_type: AccessType
    disabled: bool | None = None

    # Constructor
    def __init__(self, usr: str | None, inst: str, access: str, email: str) -> None:
        super().__init__(user_id=usr, institution=inst, access_type=access, email=email)

    def is_datakinder(self) -> bool:
        """Whether a given user is a Datakinder."""
        return self.access_type == AccessType.DATAKINDER

    def is_model_owner(self) -> bool:
        """Whether a given user is a model owner."""
        return self.access_type == AccessType.MODEL_OWNER

    def is_data_owner(self) -> bool:
        """Whether a given user is a data owner."""
        return self.access_type == AccessType.DATA_OWNER

    def is_viewer(self) -> bool:
        """Whether a given user is a viewer."""
        return self.access_type == AccessType.VIEWER

    def has_access_to_inst(self, inst: str) -> bool:
        """Whether a given user has access to a given institution."""
        return self.institution == inst or self.access_type == AccessType.DATAKINDER

    def has_full_data_access(self) -> bool:
        """Datakinders, model_owners, data_owners, all have full data access."""
        return self.access_type in (
            AccessType.DATAKINDER,
            AccessType.MODEL_OWNER,
            AccessType.DATA_OWNER,
        )

    def has_stronger_permissions_than(self, other_access_type: AccessType) -> bool:
        """Check that self has stronger permissions than other."""
        if self.access_type == AccessType.DATAKINDER:
            return True
        if self.access_type == AccessType.MODEL_OWNER:
            return other_access_type in (
                AccessType.MODEL_OWNER,
                AccessType.DATA_OWNER,
                AccessType.VIEWER,
            )
        if self.access_type == AccessType.DATA_OWNER:
            return other_access_type in (AccessType.DATA_OWNER, AccessType.VIEWER)
        if self.access_type == AccessType.VIEWER:
            return other_access_type == AccessType.VIEWER
        return False


def get_user(sess: Session, username: str) -> BaseUser:
    query_result = sess.execute(
        select(AccountTable).where(
            AccountTable.email == username,
        )
    ).all()
    if len(query_result) == 0 or len(query_result) > 1:
        return None
    return BaseUser(
        usr=uuid_to_str(query_result[0][0].id),
        inst=uuid_to_str(query_result[0][0].inst_id),
        access=query_result[0][0].access_type,
        email=username,
    )


def authenticate_user(username: str, password: str, sess: Session) -> BaseUser:
    query_result = sess.execute(
        select(AccountTable).where(
            AccountTable.email == username,
        )
    ).all()
    if len(query_result) == 0 or len(query_result) > 1:
        return False
    if not verify_password(password, query_result[0][0].password_hash):
        return False
    return BaseUser(
        usr=uuid_to_str(query_result[0][0].id),
        inst=uuid_to_str(query_result[0][0].inst_id),
        access=query_result[0][0].access_type,
        email=username,
    )


async def get_current_user(
    sess: Annotated[Session, Depends(get_session)],
    token: Annotated[str, Depends(oauth2_scheme)],
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except InvalidTokenError:
        raise credentials_exception
    user = get_user(sess, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(
    current_user: Annotated[BaseUser, Depends(get_current_user)],
):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


def has_access_to_inst_or_err(inst: str, user: BaseUser):
    """Raise error if a given user does not have access to a given institution."""
    if not user.has_access_to_inst(inst):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authorized to read this institution's resources.",
        )


def has_full_data_access_or_err(user: BaseUser, resource_type: str):
    """Raise error if a given user does not have data access to a given institution."""
    if not user.has_full_data_access():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authorized to view " + resource_type + " for this institution.",
        )


def model_owner_and_higher_or_err(user: BaseUser, resource_type: str):
    """Raise error if a given user does not have model ownership or higher."""
    if not user.access_type in (AccessType.MODEL_OWNER, AccessType.DATAKINDER):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No permissions for " + resource_type + " for this institution.",
        )


# At this point the value should not be empty as we checked on app startup.
def prepend_env_prefix(name: str) -> str:
    return env_vars["ENV"] + "_" + name


def uuid_to_str(uuid_val: uuid.UUID) -> str:
    if uuid_val is None:
        return ""
    return uuid_val.hex


def str_to_uuid(hex_str: str) -> uuid.UUID:
    return uuid.UUID(hex_str)
