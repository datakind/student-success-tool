"""Test file for the users.py file and constituent API functions.
"""

from fastapi.testclient import TestClient
from fastapi import HTTPException
import pytest
from datetime import datetime
import uuid
import sqlalchemy
from sqlalchemy.pool import StaticPool

from ..test_helper import USR, USER_ACCT_REQUEST, USER_ACCT, DATAKINDER
from .users import router
from ..main import app
from ..database import AccountTable, InstTable, Base, get_session, local_session

from ..utilities import uuid_to_str, get_current_active_user

DATETIME_TESTING = datetime(2024, 12, 24, 20, 22, 20, 132022)
UUID_1 = uuid.UUID("64dbce41-111b-46fe-8e84-c38757477ef2")
USER_UUID = uuid.UUID("5301a352-c03d-4a39-beec-16c5668c4700")
USER_VALID_INST_UUID = uuid.UUID("1d7c75c3-3eda-4294-9c66-75ea8af97b55")
INVALID_UUID = uuid.UUID("27316b89-5e04-474a-9ea4-97beaf72c9af")


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
        id=UUID_1,
        inst_id=USER_VALID_INST_UUID,
        name="John Smith",
        email="johnsmith@example.com",
        email_verified_at=None,
        password="xxxx",
        access_type="DATAKINDER",
        created_at=DATETIME_TESTING,
        updated_at=DATETIME_TESTING,
    )
    user_2 = AccountTable(
        id=USER_UUID,
        inst_id=USER_VALID_INST_UUID,
        name="Jane Doe",
        email="janedoe@example.com",
        email_verified_at=None,
        password="xxxx",
        access_type="DATAKINDER",
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
                        created_at=DATETIME_TESTING,
                        updated_at=DATETIME_TESTING,
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

    def get_current_active_user_override():
        return USR

    app.include_router(router)
    app.dependency_overrides[get_session] = get_session_override
    app.dependency_overrides[get_current_active_user] = get_current_active_user_override

    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


@pytest.fixture(name="datakinder_client")
def datakinder_client_fixture(session: sqlalchemy.orm.Session):
    def get_session_override():
        return session

    def get_current_active_user_override():
        return DATAKINDER

    app.include_router(router)
    app.dependency_overrides[get_session] = get_session_override
    app.dependency_overrides[get_current_active_user] = get_current_active_user_override

    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


def test_read_inst_users(client: TestClient):
    """Test GET /institutions/<uuid>/users."""
    response = client.get(
        "/institutions/" + uuid_to_str(USER_VALID_INST_UUID) + "/users"
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
        + uuid_to_str(UUID_1)
    )
    assert response.status_code == 200
    assert response.json() == {
        "user_id": "64dbce41111b46fe8e84c38757477ef2",
        "name": "",
        "inst_id": "1d7c75c33eda42949c6675ea8af97b55",
        "access_type": "DATAKINDER",
        "email": "",
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


def test_create_user_disallowed(client: TestClient):
    """Test POST /institutions/<uuid>/users/. For various user access types."""
    # Unauthorized.
    response = client.post(
        "/institutions/" + uuid_to_str(USER_VALID_INST_UUID) + "/users",
        json=USER_ACCT_REQUEST,
    )
    assert response.status_code == 401
    assert response.json() == {
        "detail": "Not authorized to create a more powerful user."
    }


def test_create_user_datakinder(datakinder_client: TestClient):
    # Authorized.
    response = datakinder_client.post(
        "/institutions/" + uuid_to_str(USER_VALID_INST_UUID) + "/users",
        json=USER_ACCT_REQUEST,
    )
    assert response.status_code == 200
    assert response.json() == {
        "user_id": "",
        "name": "Taylor Smith",
        "inst_id": "1d7c75c33eda42949c6675ea8af97b55",
        "access_type": "DATAKINDER",
        "email": "abc@example.com",
    }
