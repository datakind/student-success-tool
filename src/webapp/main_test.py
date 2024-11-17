"""Test file for the main.py file and constituent API functions.
"""

from fastapi.testclient import TestClient

from .main import app
from .test_helper import USR_STR

client = TestClient(app, root_path="/api/v1")


def test_read_all_inst():
    """Test GET /institutions."""
    response = client.get("/institutions" + USR_STR)
    assert response.status_code == 200
    assert response.json() == []


def test_read_inst():
    """Test GET /institutions/123. For various user access types."""
    # Unauthorized.
    response = client.get("/institutions/123" + USR_STR)
    assert response.status_code == 401
    assert response.json() == {
        "detail": "Not authorized to read this institution's resources."
    }

    # Authorized.
    response = client.get("/institutions/345" + USR_STR)
    assert response.status_code == 200
    assert response.json() == {
        "description": "",
        "inst_id": 345,
        "name": "foo",
        "retention": 0,
    }
