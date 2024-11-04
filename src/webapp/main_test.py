"""Test file for the main.py file and constituent API functions.
"""

from fastapi.testclient import TestClient

from .main import app

client = TestClient(app)


def test_read_all_inst():
    response = client.get("/institutions")
    assert response.status_code == 200
    assert response.json() == {
        "id": "foo",
        "title": "Foo",
        "description": "There goes my hero",
    }


def test_read_inst():
    response = client.get("/institutions/123")
    assert response.status_code == 404
    assert response.json() == {"detail": "Item not found"}
