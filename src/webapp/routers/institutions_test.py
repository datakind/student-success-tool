"""Test file for the institutions.py file and constituent API functions.
"""

from fastapi.testclient import TestClient
from fastapi import HTTPException
import pytest

from .institutions import router
from ..test_helper import (
    USR_STR,
    INSTITUTION_OBJ,
    VIEWER_STR,
    DATAKINDER_STR,
    EMPTY_INSTITUTION_OBJ,
)

client = TestClient(router, root_path="")


def test_read_all_inst():
    """Test GET /institutions."""
    # Unauthorized.
    with pytest.raises(HTTPException) as err:
        client.get("/institutions" + USR_STR)
    assert err.value.status_code == 401
    assert (
        err.value.detail
        == "Not authorized to read this resource. Select a specific institution."
    )
    # Authorized.
    response = client.get("/institutions" + DATAKINDER_STR)
    assert response.status_code == 200
    assert response.json() == []


def test_read_inst():
    """Test GET /institutions/123. For various user access types."""
    # Unauthorized.
    with pytest.raises(HTTPException) as err:
        client.get("/institutions/123" + USR_STR)
    assert err.value.status_code == 401
    assert err.value.detail == "Not authorized to read this institution's resources."

    # Authorized.
    response = client.get("/institutions/345" + USR_STR)
    assert response.status_code == 200
    assert response.json() == EMPTY_INSTITUTION_OBJ


def test_create_inst():
    """Test POST /institutions. For various user access types."""
    # Unauthorized.
    with pytest.raises(HTTPException) as err:
        client.post("/institutions/" + USR_STR, json=INSTITUTION_OBJ)
    assert err.value.status_code == 401
    assert err.value.detail == "Not authorized to create an institution."

    # Authorized.
    response = client.post(
        "/institutions/" + DATAKINDER_STR, json=INSTITUTION_OBJ
    )
    assert response.status_code == 200
    assert response.json() == INSTITUTION_OBJ
