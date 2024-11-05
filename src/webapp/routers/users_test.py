"""Test file for the users.py file and constituent API functions.
"""

from fastapi.testclient import TestClient
from fastapi import HTTPException
import pytest

from ..utilities import AccessType, BaseUser
from .users import router

client = TestClient(router, root_path="/api/v1")

usr = BaseUser(12, 345, AccessType.MODEL_OWNER)
usr_str = usr.construct_query_param_string()

viewer = BaseUser(12, 345, AccessType.VIEWER)
viewer_str = viewer.construct_query_param_string()

# Test GET /institutions/345/users.
def test_read_inst_users():
    response = client.get("/institutions/345/users"+usr_str)
    assert response.status_code == 200
    assert response.json() == []

# Test GET /institutions/345/users/10. For various user access types.
def test_read_inst_user():
    # Authorized.
    response = client.get("/institutions/345/users/12"+usr_str)
    assert response.status_code == 200
    assert response.json() ==  {
        "user_id": 12, 
        "name":"", 
        "inst_id":345, 
        "access_type":1, 
        "email":"", 
        "username":"", 
        "account_disabled":False, 
        "deletion_request":None 
    }
    # Unauthorized cases.
    with pytest.raises(HTTPException) as err:
        client.get("/institutions/345/users/34"+viewer_str)
    assert err.value.status_code == 401
    assert err.value.detail == "Not authorized to view this user."

    with pytest.raises(HTTPException) as err:
        client.get("/institutions/123/users/34"+usr_str)
    assert err.value.status_code == 401
    assert err.value.detail == "Not authorized to read this institution's resources."