"""Test file for the models.py file and constituent API functions.
"""

from fastapi.testclient import TestClient
from fastapi import HTTPException
import pytest

from ..test_helper import USR_STR, VIEWER_STR, INSTITUTION_OBJ
from .models import router

client = TestClient(router, root_path="/api/v1")

def test_read_inst_models():
    """Test GET /institutions/345/models."""
    response = client.get("/institutions/345/models"+USR_STR)
    assert response.status_code == 200
    assert response.json() == []

def test_read_inst_model():
    """Test GET /institutions/345/models/10. For various user access types."""
    # Authorized.
    response = client.get("/institutions/345/models/10"+USR_STR)
    assert response.status_code == 200
    assert response.json() == INSTITUTION_OBJ
    # Unauthorized cases.
    with pytest.raises(HTTPException) as err:
        client.get("/institutions/345/models/10"+VIEWER_STR)
    assert err.value.status_code == 401
    assert err.value.detail == "Not authorized to view this model for this institution."

def test_read_inst_model_versions():
    """Test GET /institutions/345/models/10/vers."""
    # Authorized.
    response = client.get("/institutions/345/models/10/vers"+USR_STR)
    assert response.status_code == 200
    assert response.json() == []

def test_read_inst_model_version():
    """Test GET /institutions/345/models/10/vers/0."""
    # Authorized.
    response = client.get("/institutions/345/models/10/vers/0"+USR_STR)
    assert response.status_code == 200
    assert response.json() == INSTITUTION_OBJ

def test_read_inst_model_outputs():
    """Test GET /institutions/345/models/10/vers/0/output."""
    # Authorized.
    response = client.get("/institutions/345/models/10/vers/0/output"+USR_STR)
    assert response.status_code == 200
    assert response.json() == []

def test_read_inst_model_output():
    """Test GET /institutions/345/models/10/vers/0/output/1."""
    # Authorized.
    response = client.get("/institutions/345/models/10/vers/0/output/1"+USR_STR)
    assert response.status_code == 200
    assert response.json() ==  {
        "m_id" :10,
        "vers_id": 0, 
        "output_id": 1,
        "executor": 123,  
        "execution_disabled": False, 
        "deletion_request": None 
    }
