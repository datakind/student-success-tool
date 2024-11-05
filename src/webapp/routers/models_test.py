"""Test file for the models.py file and constituent API functions.
"""

from fastapi.testclient import TestClient
from fastapi import HTTPException
import pytest

from ..utilities import AccessType, BaseUser
from .models import router

client = TestClient(router, root_path="/api/v1")

usr = BaseUser(12, 345, AccessType.MODEL_OWNER)
usr_str = usr.construct_query_param_string()

viewer = BaseUser(12, 345, AccessType.VIEWER)
viewer_str = viewer.construct_query_param_string()

# Test GET /institutions/345/models.
def test_read_inst_models():
    response = client.get("/institutions/345/models"+usr_str)
    assert response.status_code == 200
    assert response.json() == []

# Test GET /institutions/345/models/10. For various user access types.
def test_read_inst_model():
    # Authorized.
    response = client.get("/institutions/345/models/10"+usr_str)
    assert response.status_code == 200
    assert response.json() ==  {
        "m_id" :10,
        "name": "foo-model", 
        "vers_id": 0, 
        "description": "some model for foo", 
        "creator": 123,  
        "model_disabled": False, 
        "deletion_request": None 
    }
    # Unauthorized cases.
    with pytest.raises(HTTPException) as err:
        client.get("/institutions/345/models/10"+viewer_str)
    assert err.value.status_code == 401
    assert err.value.detail == "Not authorized to view this model for this institution."

# Test GET /institutions/345/models/10/vers.
def test_read_inst_model_versions():
    # Authorized.
    response = client.get("/institutions/345/models/10/vers"+usr_str)
    assert response.status_code == 200
    assert response.json() ==  []

# Test GET /institutions/345/models/10/vers/0.
def test_read_inst_model_version():
    # Authorized.
    response = client.get("/institutions/345/models/10/vers/0"+usr_str)
    assert response.status_code == 200
    assert response.json() ==  {
        "m_id" :10,
        "name": "foo-model", 
        "vers_id": 0, 
        "description": "some model for foo", 
        "creator": 123,  
        "model_disabled": False, 
        "deletion_request": None 
    }

# Test GET /institutions/345/models/10/vers/0/output.
def test_read_inst_model_outputs():
    # Authorized.
    response = client.get("/institutions/345/models/10/vers/0/output"+usr_str)
    assert response.status_code == 200
    assert response.json() ==  []

# Test GET /institutions/345/models/10/vers/0/output/1.
def test_read_inst_model_output():
    # Authorized.
    response = client.get("/institutions/345/models/10/vers/0/output/1"+usr_str)
    assert response.status_code == 200
    assert response.json() ==  {
        "m_id" :10,
        "vers_id": 0, 
        "output_id": 1,
        "executor": 123,  
        "execution_disabled": False, 
        "deletion_request": None 
    }
