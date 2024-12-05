"""Test file for the data.py file and constituent API functions.
"""

from fastapi.testclient import TestClient
from fastapi import HTTPException
import pytest

from ..test_helper import USR_STR, VIEWER_STR, DATA_OBJ, BATCH_REQUEST
from .data import router

client = TestClient(router, root_path="")


def test_read_inst_training_inputs():
    """Test GET /institutions/345/input_train."""
    response = client.get("/institutions/345/input_train" + USR_STR)
    assert response.status_code == 200
    assert response.json() == []


def test_read_inst_training_input():
    """Test GET /institutions/345/input/123. For various user access types."""
    # Authorized.
    response = client.get("/institutions/345/input_train/10" + USR_STR)
    assert response.status_code == 200
    assert response.json() == DATA_OBJ
    # Unauthorized cases.
    with pytest.raises(HTTPException) as err:
        client.get("/institutions/345/input_train/10" + VIEWER_STR)
    assert err.value.status_code == 401
    assert err.value.detail == "Not authorized to view input data for this institution."


def read_inst_inference_inputs():
    """Test GET /institutions/345/input_inference."""
    # Authorized.
    response = client.get("/institutions/345/input_inference/10" + USR_STR)
    assert response.status_code == 200
    assert response.json() == []


def read_inst_inference_input():
    """Test GET /institutions/345/input_inference/10."""
    # Authorized.
    response = client.get("/institutions/345/input_inference/10" + USR_STR)
    assert response.status_code == 200
    assert response.json() == DATA_OBJ


def test_create_batch():
    """Test POST /institutions/345/input_train/."""
    # Authorized.
    response = client.post(
        "/institutions/345/input_train/" + USR_STR, json=BATCH_REQUEST
    )
    assert response.status_code == 200


def test_pull_pdp_sftp():
    """Test POST /institutions/345/input_train/pdp_sftp/."""
    # Authorized.
    response = client.post("/institutions/345/input_train/10/pdp_sftp/" + USR_STR)
    assert response.status_code == 200


def test_upload():
    """Test POST /institutions/345/input_train/."""
    # Authorized.
    response = client.post("/institutions/345/input_train/10/" + USR_STR)
    assert response.status_code == 200


def test_create_batch_inference():
    """Test POST /institutions/345/input_train/pdp_sftp/."""
    # Authorized.
    response = client.post(
        "/institutions/345/input_inference/" + USR_STR, json=BATCH_REQUEST
    )
    assert response.status_code == 200


def test_pull_pdp_sftp_inference():
    """Test POST /institutions/345/input_train/pdp_sftp/."""
    # Authorized.
    response = client.post("/institutions/345/input_inference/10/" + USR_STR)
    assert response.status_code == 200


def test_upload_inference():
    """Test POST /institutions/345/input_train/."""
    # Authorized.
    response = client.post("/institutions/345/input_inference/10/pdp_sftp/" + USR_STR)
    assert response.status_code == 200
