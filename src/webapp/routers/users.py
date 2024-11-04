"""API functions related to users.
"""


from typing import Annotated
from fastapi import HTTPException, status, APIRouter
from ..utilities import has_access_to_inst_or_err, has_full_data_access_or_err, BaseUser

router = APIRouter(
    prefix="/institutions",
    tags=["users"],
)

# User account related operations.
    
@router.get("/{inst_id}/users")
def read_inst_users(
    current_user: BaseUser,
):
    """Returns all users attributed to a given institution and account type.
    
    Only visible to data owners of that institution or higher.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    has_full_data_access_or_err(current_user, "users")
    return ""

@router.get("/{inst_id}/users/{user_id}")
def read_inst_user(
    current_user: BaseUser,
):
    """Returns info on a specific user.
    
    Only visible to data owners of that institution or higher or that specific user.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    if not has_full_data_access(current_user) and current_user.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authorized to view this user.",
        )
    return ""
