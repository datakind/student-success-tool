"""Test file for the users.py file and constituent API functions.
"""

from fastapi.testclient import TestClient
from fastapi import HTTPException
import pytest

from ..test_helper import USR_STR, VIEWER_STR
from .users import router

client = TestClient(router, root_path="/api/v1")


def test_read_inst_users():
    """Test GET /institutions/345/users."""
    response = client.get("/institutions/345/users" + USR_STR)
    assert response.status_code == 200
    assert response.json() == []


def test_read_inst_user():
    """Test GET /institutions/345/users/10. For various user access types."""
    # Authorized.
    response = client.get("/institutions/345/users/12" + USR_STR)
    assert response.status_code == 200
    assert response.json() == {
        "user_id": 12,
        "name": "",
        "inst_id": 345,
        "access_type": 1,
        "email": "",
        "username": "",
        "account_disabled": False,
        "deletion_request": None,
    }
    # Unauthorized cases.
    with pytest.raises(HTTPException) as err:
        client.get("/institutions/345/users/34" + VIEWER_STR)
    assert err.value.status_code == 401
    assert err.value.detail == "Not authorized to view this user."

    with pytest.raises(HTTPException) as err:
        client.get("/institutions/123/users/34" + USR_STR)
    assert err.value.status_code == 401
    assert err.value.detail == "Not authorized to read this institution's resources."
