"""Test file for the users.py file and constituent API functions.
"""

from fastapi.testclient import TestClient
from fastapi import HTTPException
import pytest
from datetime import datetime
import uuid
import sqlalchemy
from sqlalchemy.pool import StaticPool

from ..test_helper import (
    USR,
    VIEWER_STR,
    USER_ACCT_REQUEST,
    USER_ACCT,
    DATAKINDER_STR,
    USR_STR,
    USER_UUID,
    USER_VALID_INST_UUID,
    UUID_INVALID,
    USER_1_UUID,
    DATETIME_TESTING,
    USER_ACCT,
)
from .users import router
from ..main import app
from ..database import AccountTable, InstTable, Base, get_session, local_session

from ..utilities import uuid_to_str


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
        id=USER_1_UUID,
        inst_id=USER_VALID_INST_UUID,
        name="John Smith",
        email="johnsmith@example.com",
        email_verified=True,
        password_hash="xxxx",
        access_type="DATAKINDER",
        time_created=DATETIME_TESTING,
        time_updated=DATETIME_TESTING,
    )
    user_2 = AccountTable(
        id=USER_UUID,
        inst_id=USER_VALID_INST_UUID,
        name="Taylor Smith",
        email="abc@example.com",
        email_verified=True,
        password_hash="xxxx",
        access_type="DATAKINDER",
        time_created=DATETIME_TESTING,
        time_updated=DATETIME_TESTING,
    )
    try:
        with sqlalchemy.orm.Session(engine) as session:
            session.add_all(
                [
                    InstTable(
                        id=USER_VALID_INST_UUID,
                        name="school_1",
                        time_created=DATETIME_TESTING,
                        time_updated=DATETIME_TESTING,
                    ),
                    user_1,
                    user_2,
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


def test_read_inst_users(client: TestClient):
    """Test GET /institutions/<uuid>/users."""
    response = client.get(
        "/institutions/" + uuid_to_str(USER_VALID_INST_UUID) + "/users" + USR_STR
    )
    assert response.status_code == 200
    assert response.json() == []


def test_read_inst_user(client: TestClient):
    """Test GET /institutions/<uuid>/users. For various user access types."""
    # Authorized.
    response = client.get(
        "/institutions/"
        + uuid_to_str(USER_VALID_INST_UUID)
        + "/users/"
        + uuid_to_str(USER_UUID)
        + USR_STR
    )
    assert response.status_code == 200
    assert response.json() == {
        "access_type": "DATAKINDER",
        "account_disabled": False,
        "deletion_request": None,
        "email": "",
        "inst_id": "1d7c75c33eda42949c6675ea8af97b55",
        "name": "",
        "user_id": "5301a352c03d4a39beec16c5668c4700",
    }
    # Unauthorized cases.
    """
    with pytest.raises(HTTPException) as err:
        client.get("/institutions/"+ uuid_to_str(USER_VALID_INST_UUID) +"/users/34" + VIEWER_STR)
    assert err.value.status_code == 401
    assert err.value.detail == "Not authorized to view this user."

    with pytest.raises(HTTPException) as err:
        client.get("/institutions/"+ uuid_to_str(USER_VALID_INST_UUID) +"/users/34" + USR_STR)
    assert err.value.status_code == 401
    assert err.value.detail == "Not authorized to read this institution's resources."
    """


def test_create_user(client: TestClient):
    """Test POST /institutions/<uuid>/users/. For various user access types."""
    # Unauthorized.
    response = client.post(
        "/institutions/" + uuid_to_str(USER_VALID_INST_UUID) + "/users" + VIEWER_STR,
        json=USER_ACCT_REQUEST,
    )
    assert response.status_code == 401

    # Authorized.
    response = client.post(
        "/institutions/"
        + uuid_to_str(USER_VALID_INST_UUID)
        + "/users"
        + DATAKINDER_STR,
        json=USER_ACCT_REQUEST,
    )
    assert response.status_code == 200
