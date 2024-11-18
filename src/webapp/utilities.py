"""Helper functions that may be used across multiple API router subpackages.
"""

from typing import Union
from enum import IntEnum

from fastapi import HTTPException, status
from pydantic import BaseModel, ConfigDict


# TODO: Store in a python package to be usable by the frontend.
class AccessType(IntEnum):
    """Access types available."""

    DATAKINDER = 1
    MODEL_OWNER = 2
    DATA_OWNER = 3
    VIEWER = 4


class BaseUser(BaseModel):
    """BaseUser represents an access type. The frontend will include more detailed User info."""

    model_config = ConfigDict(use_enum_values=True)
    # user_id is permanent and each frontend orginated account will map to a unique user_id.
    # Bare API callers will likely not include a user_id.
    user_id: Union[int, None] = None
    # For Datakinders, institution = 0 (reserved value) which means "no inst specified".
    institution: int
    access_type: AccessType

    # Constructor
    def __init__(self, usr: Union[int, None], inst: int, access: AccessType) -> None:
        super().__init__(user_id=usr, institution=inst, access_type=access)

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

    def has_access_to_inst(self, inst: int) -> bool:
        """Whether a given user has access to a given institution."""
        return self.institution == inst or self.access_type == AccessType.DATAKINDER

    def has_full_data_access(self) -> bool:
        """Datakinders, model_owners, data_owners, all have full data access."""
        return self.access_type in (
            AccessType.DATAKINDER,
            AccessType.MODEL_OWNER,
            AccessType.DATA_OWNER,
        )

    def construct_query_param_string(self) -> str:
        """Construct query paramstring from BaseUser. Mostly used for testing."""
        ret = "?"
        if self.user_id is not None:
            ret += "usr=" + str(self.user_id) + "&"
        ret += "inst=" + str(self.institution) + "&"
        ret += "access=" + str(self.access_type)
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


def has_access_to_inst_or_err(inst: int, user: BaseUser):
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
