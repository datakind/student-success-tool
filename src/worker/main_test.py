"""Test file for the main.py file and constituent API functions.
"""

import pytest
import json

from fastapi.testclient import TestClient
from .main import app
import uuid
from .authn import get_current_username
from unittest import mock
from .utilities import StorageControl

MOCK_STORAGE = mock.Mock()
MOCK_BACKEND_API = mock.Mock()


@pytest.fixture(name="client")
def client_fixture():
    def get_current_username_override():
        return "testing_username"

    def storage_control_override():
        return MOCK_STORAGE

    app.dependency_overrides[StorageControl] = storage_control_override

    app.dependency_overrides[get_current_username] = get_current_username_override

    client = TestClient(app, root_path="/workers/api/v1")
    yield client
    app.dependency_overrides.clear()


def test_get_root(client: TestClient):
    """Test GET /."""
    response = client.get("/")
    assert response.status_code == 200


def test_retrieve_token(client: TestClient):
    """Test POST /token."""
    response = client.post(
        "/token",
        data={"username": "tester-user", "password": "tester-pw"},
        headers={"content-type": "application/x-www-form-urlencoded"},
    )
    assert response.status_code == 200


def test_execute_pdp_pull(client: TestClient):
    """Test POST /execute-pdp-pull."""
    MOCK_STORAGE.copy_from_sftp_to_gcs.return_value = None
    MOCK_STORAGE.create_bucket_if_not_exists.return_value = None
    MOCK_BACKEND_API.get_pdp_id.return_value = {
        "inst_id": "942d4b0e12e74d2a91879508ae3cef7c",
        "name": "University of South Carolina - Beaufort",
        "state": "SC",
        "retention_days": None,
        "pdp_id": "345000"
        }
  
    response = client.post("/execute-pdp-pull", json={"placeholder": "val"})
    assert response.status_code == 200
    assert response.json() == {
        "pdp_inst_generated": [],
        "pdp_inst_not_found": [],
    }
