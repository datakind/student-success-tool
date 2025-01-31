"""Test file for the institutions.py file and constituent API functions.
"""

from fastapi.testclient import TestClient
from fastapi import HTTPException
import pytest
from datetime import datetime
import sqlalchemy
from sqlalchemy.pool import StaticPool
import uuid
import os
from unittest import mock

from . import institutions
from ..test_helper import (
    INSTITUTION_REQ,
    INSTITUTION_REQ_BAREBONES,
    INSTITUTION_OBJ,
    USR,
    DATAKINDER,
)

from ..utilities import uuid_to_str, get_current_active_user
from ..main import app
from ..database import InstTable, Base, get_session, local_session
from ..gcsutil import StorageControl

DATETIME_TESTING = datetime.today()
UUID_1 = uuid.uuid4()
UUID_2 = uuid.uuid4()
USER_UUID = uuid.UUID("5301a352-c03d-4a39-beec-16c5668c4700")
USER_VALID_INST_UUID = uuid.UUID("1d7c75c3-3eda-4294-9c66-75ea8af97b55")
INVALID_UUID = uuid.UUID("27316b89-5e04-474a-9ea4-97beaf72c9af")

MOCK_STORAGE = mock.AsyncMock()


@pytest.fixture(name="session")
def session_fixture():
    engine = sqlalchemy.create_engine(
        "sqlite://",
        echo=True,
        echo_pool="debug",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    try:
        with sqlalchemy.orm.Session(engine) as session:
            session.add_all(
                [
                    InstTable(
                        id=UUID_1,
                        name="school_1",
                        created_at=DATETIME_TESTING,
                        updated_at=DATETIME_TESTING,
                    ),
                    InstTable(
                        id=UUID_2,
                        name="school_2",
                        created_at=DATETIME_TESTING,
                        updated_at=DATETIME_TESTING,
                    ),
                    InstTable(
                        id=USER_VALID_INST_UUID,
                        name="valid_school",
                        created_at=DATETIME_TESTING,
                        updated_at=DATETIME_TESTING,
                    ),
                ]
            )
            session.commit()
            yield session
    finally:
        Base.metadata.drop_all(engine)


@pytest.fixture(name="client")
def client_fixture(session: sqlalchemy.orm.Session):
    def get_session_override():
        return session

    def get_current_active_user_override():
        return USR

    def storage_control_override():
        return MOCK_STORAGE

    app.include_router(institutions.router)
    app.dependency_overrides[get_session] = get_session_override
    app.dependency_overrides[get_current_active_user] = get_current_active_user_override
    app.dependency_overrides[StorageControl] = storage_control_override

    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


@pytest.fixture(name="datakinder_client")
def datakinder_client_fixture(session: sqlalchemy.orm.Session):
    def get_session_override():
        return session

    def get_current_active_user_override():
        return DATAKINDER

    def storage_control_override():
        return MOCK_STORAGE

    app.include_router(institutions.router)
    app.dependency_overrides[get_session] = get_session_override
    app.dependency_overrides[get_current_active_user] = get_current_active_user_override
    app.dependency_overrides[StorageControl] = storage_control_override

    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


def test_read_all_inst(client: TestClient):
    """Test GET /institutions."""

    # Unauthorized.
    response = client.get("/institutions")
    assert str(response) == "<Response [401 Unauthorized]>"
    assert (
        response.text
        == '{"detail":"Not authorized to read this resource. Select a specific institution."}'
    )


def test_read_all_inst_datakinder(datakinder_client: TestClient):
    """Test GET /institutions."""
    # Authorized.
    response = datakinder_client.get("/institutions")
    assert response.status_code == 200
    assert response.json() == [
        {
            "description": None,
            "inst_id": uuid_to_str(UUID_1),
            "name": "school_1",
            "retention_days": None,
        },
        {
            "description": None,
            "inst_id": uuid_to_str(UUID_2),
            "name": "school_2",
            "retention_days": None,
        },
        {
            "description": None,
            "inst_id": uuid_to_str(USER_VALID_INST_UUID),
            "name": "valid_school",
            "retention_days": None,
        },
    ]


def test_read_inst_by_name(client: TestClient):
    # Test GET /institutions/<uuid>. For various user access types.
    # Unauthorized.
    response = client.get("/institutions/name/school_1")

    assert str(response) == "<Response [401 Unauthorized]>"
    assert (
        response.text
        == '{"detail":"Not authorized to read this institution\'s resources."}'
    )

    # Authorized.
    response = client.get("/institutions/name/valid_school")
    assert response.status_code == 200
    assert response.json() == INSTITUTION_OBJ


def test_read_inst(client: TestClient):
    # Test GET /institutions/<uuid>. For various user access types.
    # Unauthorized.
    response = client.get("/institutions/" + uuid_to_str(UUID_1))

    assert str(response) == "<Response [401 Unauthorized]>"
    assert (
        response.text
        == '{"detail":"Not authorized to read this institution\'s resources."}'
    )

    # Authorized.
    response = client.get("/institutions/" + uuid_to_str(USER_VALID_INST_UUID))
    assert response.status_code == 200
    assert response.json() == INSTITUTION_OBJ


def test_create_inst_unauth(client):
    # Test POST /institutions. For various user access types.
    os.environ["ENV"] = "DEV"
    # Unauthorized.
    response = client.post("/institutions", json=INSTITUTION_REQ)
    assert str(response) == "<Response [401 Unauthorized]>"
    assert response.text == '{"detail":"Not authorized to create an institution."}'


def test_create_inst(datakinder_client):
    # Test POST /institutions. For various user access types.
    os.environ["ENV"] = "DEV"
    assert "DEV" == os.environ.get("ENV")
    MOCK_STORAGE.create_bucket.return_value = None
    MOCK_STORAGE.create_folders.return_value = None
    # Authorized.
    response = datakinder_client.post("/institutions", json=INSTITUTION_REQ)
    assert response.status_code == 200
    assert response.json()["name"] == "foobar school"
    assert response.json()["description"] == "description of school"
    assert response.json()["retention_days"] == 1
    assert response.json()["inst_id"] != None

    response = datakinder_client.post("/institutions", json=INSTITUTION_REQ_BAREBONES)
    assert response.status_code == 200
    assert response.json()["name"] == "testing school"
