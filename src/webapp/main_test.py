"""Test file for the main.py file and constituent API functions.
"""

from fastapi.testclient import TestClient

from .main import app

client = TestClient(app, root_path="/api/v1")


def test_get_root():
    """Test GET /."""
    response = client.get("/")
    assert response.status_code == 200
