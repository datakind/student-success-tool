"""Test file for the main.py file and constituent API functions.
"""

from fastapi.testclient import TestClient

from .main import app

client = TestClient(app)



# xxx wip
def test_read_all_inst():
    response = client.get("/institutions", headers={}, json={
            "current_user": "foo",
        },
    )
    assert response.status_code == 200
    assert response.json() == {"message": "All instutions"}


def test_read_inst():
    response = client.get("/institutions/123", headers={}, json={
            "current_user": "foo",
        },
    )
    assert response.status_code == 200
    assert response.json() == {"message": "instution 123"}
