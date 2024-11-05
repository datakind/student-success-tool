"""Test file for the main.py file and constituent API functions.
"""

from fastapi.testclient import TestClient

from .main import app
from .utilities import AccessType, BaseUser

client = TestClient(app, root_path="/api/v1")

usr = BaseUser(123, 345, AccessType.MODEL_OWNER)
usr_str = usr.construct_query_param_string()

# Test GET /institutions.
def test_read_all_inst():
    response = client.get("/institutions"+usr_str)
    assert response.status_code == 200
    assert response.json() == []

# Test GET /institutions/123. For various user access types.
def test_read_inst():
    # Unauthorized.
    response = client.get("/institutions/123"+usr_str)
    assert response.status_code == 401
    assert response.json() == {"detail": "Not authorized to read this institution's resources."}

    # Authorized.
    response = client.get("/institutions/345"+usr_str)
    assert response.status_code == 200
    assert response.json() ==  {'description': '', 'inst_id': 345, 'name': 'foo', 'retention': 0}