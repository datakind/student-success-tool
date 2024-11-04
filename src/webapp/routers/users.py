"""API functions related to users.
"""

# User account related operations.
    
@app.get("/institutions/{inst_id}/users")
def read_inst_users(
    current_user: Annotated[BaseUser],
):
    """Returns all users attributed to a given institution and account type.
    
    Only visible to data owners of that institution or higher.

    Args:
        current_user: the user making the request.
    """
    has_access_to_inst_or_err(inst_id, current_user)
    has_full_data_access_or_err(current_user, "users")
    return ""

@app.get("/institutions/{inst_id}/users/{user_id}")
def read_inst_user(
    current_user: Annotated[BaseUser],
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
