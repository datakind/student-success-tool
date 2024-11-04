"""Helper functions that may be used across multiple API router subpackages.
"""
from typing import Annotated, Union
from enum import Enum

from fastapi import HTTPException, status
from pydantic import BaseModel

# TODO: Store in a python package to be usable by the frontend.
# Accesstypes in order of decreasing access.
class AccessType(Enum):
    DATAKINDER = 1
    MODEL_OWNER = 2
    DATA_OWNER = 3
    VIEWER = 4

# BaseUser represents an access type. The frontend will include more detailed User info.
class BaseUser(BaseModel):
    # user_id is permanent and each frontend orginated account will map to a unique user_id.
    # Bare API callers will likely not include a user_id.
    user_id: Union[int, None] = None
    institution: int
    access_type: AccessType

    # Constructor
    def __init__(self, usr: Union[int, None], inst: int, access: AccessType) -> None:
        super().__init__(user_id=usr, institution=inst, access_type=access)

    # Whether a given user is a Datakinder.
    def is_datakinder(self) -> bool:
        return self.access_type == AccessType.DATAKINDER

    # Whether a given user has access to a given institution.
    def has_access_to_inst(self, inst: int) -> bool:
        return self.institution == inst or self.access_type == AccessType.DATAKINDER

    # Datakinders, model_owners, data_owners, all have full data access.
    def has_full_data_access(self) -> bool:
        return self.access_type == AccessType.DATAKINDER or self.access_type == AccessType.MODEL_OWNER or self.access_type == AccessType.DATA_OWNER

# Raise error if a given user does not have access to a given institution.
def has_access_to_inst_or_err(inst: int, user: BaseUser):
    if not user.has_access_to_inst(inst):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authorized to read this institution's resources.",
        )
    return

# Raise error if a given user does not have data access to a given institution.
def has_full_data_access_or_err(user: BaseUser, resource_type: str):
    if not user.has_full_data_access():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authorized to view " + resource_type + " for this institution.",
        )
    return
