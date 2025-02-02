"""Test file for the main.py file and constituent API functions.
"""

import pytest
import json

from fastapi.testclient import TestClient
from .main import app
import sqlalchemy
from sqlalchemy.pool import StaticPool
import uuid
from .database import AccountTable, InstTable, Base, get_session, local_session
from .authn import get_password_hash
from .test_helper import (
    DATAKINDER,
    USER_VALID_INST_UUID,
    DATETIME_TESTING,
    USER_UUID,
    USER_1_UUID,
    UUID_INVALID,
    UNASSIGNED_USER,
)
from .utilities import get_current_active_user


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
    user_1 = AccountTable(
        id=USER_UUID,
        inst_id=USER_VALID_INST_UUID,
        name="John Smith",
        email="johnsmith@example.com",
        email_verified_at=None,
        password=get_password_hash("xxxx"),
        access_type="VIEWER",
        created_at=DATETIME_TESTING,
        updated_at=DATETIME_TESTING,
    )
    user_2 = AccountTable(
        id=USER_1_UUID,
        inst_id=None,
        name="Jane Doe",
        email="janedoe@example.com",
        email_verified_at=None,
        password=get_password_hash("abc"),
        access_type="DATAKINDER",
        created_at=DATETIME_TESTING,
        updated_at=DATETIME_TESTING,
    )
    user_3 = AccountTable(
        id=UUID_INVALID,  # This says UUID_INVALID, but it is just using for convenience this UUID.
        inst_id=None,
        name="New Datakinder",
        email="new_dk@example.com",
        email_verified_at=None,
        password=get_password_hash("abc"),
        access_type=None,
        created_at=DATETIME_TESTING,
        updated_at=DATETIME_TESTING,
    )
    try:
        with sqlalchemy.orm.Session(engine) as session:
            session.add_all(
                [
                    InstTable(
                        id=USER_VALID_INST_UUID,
                        name="school_1",
                        allowed_emails=json.loads('{"jamie@example.com" : "VIEWER"}'),
                        created_at=DATETIME_TESTING,
                        updated_at=DATETIME_TESTING,
                    ),
                    user_1,
                    user_2,
                    user_3,
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
        return DATAKINDER

    app.dependency_overrides[get_session] = get_session_override
    app.dependency_overrides[get_current_active_user] = get_current_active_user_override

    client = TestClient(app, root_path="/api/v1")
    yield client
    app.dependency_overrides.clear()


@pytest.fixture(name="user_client")
def user_client_fixture(session: sqlalchemy.orm.Session):
    def get_session_override():
        return session

    def get_current_active_user_override():
        return UNASSIGNED_USER

    app.dependency_overrides[get_session] = get_session_override
    app.dependency_overrides[get_current_active_user] = get_current_active_user_override

    client = TestClient(app, root_path="/api/v1")
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
        data={"username": "johnsmith@example.com", "password": "xxxx"},
        headers={"content-type": "application/x-www-form-urlencoded"},
    )
    assert response.status_code == 200


def test_get_cross_isnt_users(client: TestClient):
    """Test GET /non_inst_users."""
    response = client.get("/non_inst_users")
    assert response.status_code == 200
    assert response.json() == [
        {
            "access_type": "DATAKINDER",
            "email": "janedoe@example.com",
            "inst_id": "",
            "name": "Jane Doe",
            "user_id": "27316b895e04474a9ea497beaf72c9af",
        },
        {
            "access_type": None,
            "email": "new_dk@example.com",
            "inst_id": "",
            "name": "New Datakinder",
            "user_id": "64dbce41111b46fe8e84c38757477ef2",
        },
    ]


def test_set_datakinders(client: TestClient):
    """Test POST /datakinders."""
    response = client.post("/datakinders", json=["new_dk@example.com"])
    assert response.status_code == 200
    assert response.json() == ["new_dk@example.com"]


def test_check_self_datakinder(client: TestClient):
    """Test GET /check_self."""
    response = client.get("/check_self")
    assert response.status_code == 200
    assert response.json() == {
        "access_type": "DATAKINDER",
        "email": "taylor@example.com",
        "inst_id": "",
        "user_id": "5301a352c03d4a39beec16c5668c4700",
    }


def test_check_self(user_client: TestClient):
    """Test GET /check_self."""
    response = user_client.get("/check_self")
    assert response.status_code == 200
    assert response.json() == {
        "access_type": "VIEWER",
        "email": "jamie@example.com",
        "inst_id": "1d7c75c33eda42949c6675ea8af97b55",
        "user_id": "e4862c62829440d8ab4c9c298f02f619",
    }
