"""Test file for the data.py file and constituent API functions.
"""

from fastapi.testclient import TestClient
from fastapi import HTTPException
import pytest

from ..utilities import AccessType, BaseUser
from .data import router

client = TestClient(router, root_path="/api/v1")

usr = BaseUser(12, 345, AccessType.MODEL_OWNER)
usr_str = usr.construct_query_param_string()

viewer = BaseUser(12, 345, AccessType.VIEWER)
viewer_str = viewer.construct_query_param_string()

# Test GET /institutions/345/input_train.
def test_read_inst_training_inputs():
    response = client.get("/institutions/345/input_train"+usr_str)
    assert response.status_code == 200
    assert response.json() == []

# Test GET /institutions/345/input/123. For various user access types.
def test_read_inst_training_input():
    # Authorized.
    response = client.get("/institutions/345/input_train/10"+usr_str)
    assert response.status_code == 200
    assert response.json() ==  {
        "batch_id" :10,
        "name": "foo-data", 
        "record_count": 100, 
        "size": 1,
        "description": "some model for foo", 
        "uploader": 123,
        "source": "MANUAL_UPLOAD",
        "data_disabled": False, 
        "deletion_request": None 
    }
    # Unauthorized cases.
    with pytest.raises(HTTPException) as err:
        client.get("/institutions/345/input_train/10"+viewer_str)
    assert err.value.status_code == 401
    assert err.value.detail == "Not authorized to view input data for this institution."

# Test GET /institutions/345/input_exec.
def read_inst_exec_inputs():
    # Authorized.
    response = client.get("/institutions/345/input_exec/10"+usr_str)
    assert response.status_code == 200
    assert response.json() ==  []

# Test GET /institutions/345/input_exec/10.
def read_inst_exec_input():
    # Authorized.
    response = client.get("/institutions/345/input_exec/10"+usr_str)
    assert response.status_code == 200
    assert response.json() ==  {
        "batch_id" :10,
        "name": "foo-data", 
        "record_count": 100, 
        "size": 1,
        "description": "some model for foo", 
        "uploader": 123,
        "source": "MANUAL_UPLOAD",
        "data_disabled": False, 
        "deletion_request": None 
    }
