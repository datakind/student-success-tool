"""Test file for the institutions.py file and constituent API functions.
"""

from fastapi.testclient import TestClient
from fastapi import HTTPException
import pytest
from datetime import datetime
import sqlalchemy
from sqlalchemy.pool import StaticPool
import uuid

from .institutions import router
from ..test_helper import (
    USR_STR,
    INSTITUTION_REQ,
    VIEWER_STR,
    DATAKINDER_STR,
    INSTITUTION_OBJ,
)
from ..main import app
from ..database import InstTable, Base, get_session, local_session

DATETIME_TESTING = datetime.today()
UUID_1 = uuid.uuid4()
UUID_2 = uuid.uuid4()
USER_UUID = "5301a352-c03d-4a39-beec-16c5668c4700"
USER_VALID_INST_UUID = "1d7c75c3-3eda-4294-9c66-75ea8af97b55"
INVALID_UUID = "27316b89-5e04-474a-9ea4-97beaf72c9af"


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
                        time_created=DATETIME_TESTING,
                        time_updated=DATETIME_TESTING,
                    ),
                    InstTable(
                        id=UUID_2,
                        name="school_2",
                        time_created=DATETIME_TESTING,
                        time_updated=DATETIME_TESTING,
                    ),
                    InstTable(
                        id=uuid.UUID(USER_VALID_INST_UUID),
                        name="valid_school",
                        time_created=DATETIME_TESTING,
                        time_updated=DATETIME_TESTING,
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

    app.include_router(router)
    app.dependency_overrides[get_session] = get_session_override

    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


def test_read_all_inst(client: TestClient):
    """Test GET /institutions."""

    # Unauthorized.
    response = client.get("/institutions" + USR_STR)
    assert str(response) == "<Response [401 Unauthorized]>"
    assert (
        response.text
        == '{"detail":"Not authorized to read this resource. Select a specific institution."}'
    )
    # Authorized.
    response = client.get("/institutions" + DATAKINDER_STR)
    assert response.status_code == 200
    assert response.json() == [
        {
            "description": None,
            "inst_id": str(UUID_1),
            "name": "school_1",
            "retention_days": None,
        },
        {
            "description": None,
            "inst_id": str(UUID_2),
            "name": "school_2",
            "retention_days": None,
        },
        {
            "description": None,
            "inst_id": "1d7c75c3-3eda-4294-9c66-75ea8af97b55",
            "name": "valid_school",
            "retention_days": None,
        },
    ]


def test_read_inst(client: TestClient):
    # Test GET /institutions/<uuid>. For various user access types.
    # Unauthorized.
    response = client.get("/institutions/school_1" + USR_STR)

    assert str(response) == "<Response [401 Unauthorized]>"
    assert (
        response.text
        == '{"detail":"Not authorized to read this institution\'s resources."}'
    )

    # Authorized.
    response = client.get("/institutions/valid_school" + USR_STR)
    assert response.status_code == 200
    assert response.json() == INSTITUTION_OBJ


def test_create_inst(client: TestClient):
    # Test POST /institutions. For various user access types.
    # Unauthorized.
    response = client.post("/institutions/" + USR_STR, json=INSTITUTION_REQ)
    assert str(response) == "<Response [401 Unauthorized]>"
    assert response.text == '{"detail":"Not authorized to create an institution."}'

    # Authorized.
    response = client.post("/institutions/" + DATAKINDER_STR, json=INSTITUTION_REQ)
    assert response.status_code == 200
    assert response.json()["name"] == "foobar school"
    assert response.json()["description"] == "description of school"
    assert response.json()["retention_days"] == 1
    # The following changes (computes on sql server write) so we just check that it's there.
    assert response.json()["inst_id"] != None
