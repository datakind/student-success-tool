"""Test file for the data.py file and constituent API functions.
"""

from fastapi.testclient import TestClient
from fastapi import HTTPException
import pytest

from ..test_helper import USR_STR, VIEWER_STR, BATCH_DATA_OBJ
from .data import router

client = TestClient(router, root_path="/api/v1")

def test_read_inst_training_inputs():
    """Test GET /institutions/345/input_train."""
    response = client.get("/institutions/345/input_train"+USR_STR)
    assert response.status_code == 200
    assert response.json() == []

def test_read_inst_training_input():
    """Test GET /institutions/345/input/123. For various user access types."""
    # Authorized.
    response = client.get("/institutions/345/input_train/10"+USR_STR)
    assert response.status_code == 200
    assert response.json() ==  BATCH_DATA_OBJ
    # Unauthorized cases.
    with pytest.raises(HTTPException) as err:
        client.get("/institutions/345/input_train/10"+VIEWER_STR)
    assert err.value.status_code == 401
    assert err.value.detail == "Not authorized to view input data for this institution."

def read_inst_exec_inputs():
    """Test GET /institutions/345/input_exec."""
    # Authorized.
    response = client.get("/institutions/345/input_exec/10"+USR_STR)
    assert response.status_code == 200
    assert response.json() ==  []

def read_inst_exec_input():
    """Test GET /institutions/345/input_exec/10."""
    # Authorized.
    response = client.get("/institutions/345/input_exec/10"+USR_STR)
    assert response.status_code == 200
    assert response.json() ==  BATCH_DATA_OBJ
