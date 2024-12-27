"""Helper functions that may be used across multiple API router subpackages.
"""

from typing import Union
from enum import StrEnum
import uuid
import os

from fastapi import HTTPException, status
from pydantic import BaseModel, ConfigDict


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

    # TODO: remove
    def construct_query_param_string(self) -> str:
        """Construct query paramstring from BaseUser. Mostly used for testing."""
        ret = "?"
        if self.user_id is not None:
            ret += "usr=" + self.user_id + "&"
        ret += "inst=" + self.institution + "&"
        ret += "access=" + self.access_type + "&"
        ret += "email=" + self.email
        return ret

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
    return os.environ["ENV"] + "_" + name


def uuid_to_str(uuid_val: uuid.UUID) -> str:
    if uuid_val is None:
        return ""
    return uuid_val.hex


def str_to_uuid(hex_str: str) -> uuid.UUID:
    return uuid.UUID(hex_str)
