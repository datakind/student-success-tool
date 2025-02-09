"""Test file for the models.py file and constituent API functions.
"""

from fastapi.testclient import TestClient
from fastapi import HTTPException
import pytest
import sqlalchemy
from sqlalchemy.pool import StaticPool
import uuid
import os
from unittest import mock
from ..test_helper import (
    DATA_OBJ,
    BATCH_REQUEST,
    USR,
    USER_VALID_INST_UUID,
    USER_UUID,
    UUID_INVALID,
    DATETIME_TESTING,
    MODEL_OBJ,
    SAMPLE_UUID,
)
from ..main import app
from ..database import (
    FileTable,
    BatchTable,
    InstTable,
    Base,
    get_session,
    local_session,
    ModelTable,
)
from ..utilities import uuid_to_str, get_current_active_user, SchemaType
from .models import router, ModelInfo, RunInfo
from collections import Counter
from ..gcsutil import StorageControl

MOCK_STORAGE = mock.Mock()

UUID_2 = uuid.UUID("9bcbc782-2e71-4441-afa2-7a311024a5ec")
FILE_UUID_1 = uuid.UUID("f0bb3a20-6d92-4254-afed-6a72f43c562a")
FILE_UUID_2 = uuid.UUID("cb02d06c-2a59-486a-9bdd-d394a4fcb833")
FILE_UUID_3 = uuid.UUID("fbe67a2e-50e0-40c7-b7b8-07043cb813a5")
BATCH_UUID = uuid.UUID("5b2420f3-1035-46ab-90eb-74d5df97de43")
created_by_UUID = uuid.UUID("0ad8b77c-49fb-459a-84b1-8d2c05722c4a")


# TODO plumb through schema configs
def same_model_orderless(a_elem: ModelInfo, b_elem: ModelInfo):
    if (
        a_elem["inst_id"] != b_elem["inst_id"]
        or a_elem["name"] != b_elem["name"]
        or a_elem["m_id"] != b_elem["m_id"]
        or a_elem["vers_id"] != b_elem["vers_id"]
        or a_elem["valid"] != b_elem["valid"]
        or a_elem["description"] != b_elem["description"]
        or a_elem["deleted"] != b_elem["deleted"]
    ):
        return False
    return True


def same_run_info_orderless(a_elem: ModelInfo, b_elem: ModelInfo):
    if (
        a_elem["inst_id"] != b_elem["inst_id"]
        or a_elem["name"] != b_elem["name"]
        or a_elem["m_id"] != b_elem["m_id"]
        or a_elem["vers_id"] != b_elem["vers_id"]
        or a_elem["valid"] != b_elem["valid"]
        or a_elem["description"] != b_elem["description"]
        or a_elem["deleted"] != b_elem["deleted"]
    ):
        return False
    return True


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
    batch_1 = BatchTable(
        id=BATCH_UUID,
        inst_id=USER_VALID_INST_UUID,
        name="batch_foo",
        created_by=created_by_UUID,
        created_at=DATETIME_TESTING,
        updated_at=DATETIME_TESTING,
    )
    file_1 = FileTable(
        id=FILE_UUID_1,
        inst_id=USER_VALID_INST_UUID,
        name="file_input_one",
        source="MANUAL_UPLOAD",
        batches={batch_1},
        created_at=DATETIME_TESTING,
        updated_at=DATETIME_TESTING,
        sst_generated=False,
        valid=True,
        schemas=[SchemaType.UNKNOWN],
    )
    file_3 = FileTable(
        id=FILE_UUID_3,
        inst_id=USER_VALID_INST_UUID,
        name="file_output_one",
        batches={batch_1},
        created_at=DATETIME_TESTING,
        updated_at=DATETIME_TESTING,
        sst_generated=True,
        valid=True,
        schemas=[SchemaType.PDP_COHORT],
    )
    model_1 = ModelTable(
        id=SAMPLE_UUID,
        inst_id=USER_VALID_INST_UUID,
        name="sample_model_for_school_1",
        valid=True,
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
                    batch_1,
                    file_1,
                    FileTable(
                        id=FILE_UUID_2,
                        inst_id=USER_VALID_INST_UUID,
                        name="file_input_two",
                        source="PDP_SFTP",
                        created_at=DATETIME_TESTING,
                        updated_at=DATETIME_TESTING,
                        sst_generated=False,
                        valid=False,
                        schemas=[SchemaType.PDP_COURSE],
                    ),
                    file_3,
                    model_1,
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

    app.include_router(router)
    app.dependency_overrides[get_session] = get_session_override
    app.dependency_overrides[get_current_active_user] = get_current_active_user_override
    app.dependency_overrides[StorageControl] = storage_control_override

    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


def test_read_inst_models(client: TestClient):
    """Test GET /institutions/345/models."""
    response = client.get(
        "/institutions/" + uuid_to_str(USER_VALID_INST_UUID) + "/models"
    )
    assert response.status_code == 200
    assert same_model_orderless(
        response.json()[0],
        {
            "created_by": "",
            "deleted": None,
            "description": None,
            "inst_id": "1d7c75c33eda42949c6675ea8af97b55",
            "m_id": "e4862c62829440d8ab4c9c298f02f619",
            "name": "sample_model_for_school_1",
            "valid": True,
            "vers_id": 0,
        },
    )


def test_read_inst_model(client: TestClient):
    """Test GET /institutions/345/models/10. For various user access types."""
    # Unauthorized cases.
    response_unauth = client.get(
        "/institutions/"
        + uuid_to_str(UUID_INVALID)
        + "/models/sample_model_for_school_1"
    )
    assert str(response_unauth) == "<Response [401 Unauthorized]>"
    assert (
        response_unauth.text
        == '{"detail":"Not authorized to read this institution\'s resources."}'
    )

    # Authorized.
    response = client.get(
        "/institutions/"
        + uuid_to_str(USER_VALID_INST_UUID)
        + "/models/sample_model_for_school_1"
    )
    assert response.status_code == 200
    assert same_model_orderless(response.json()[0], MODEL_OBJ)


def test_read_inst_model_version(client: TestClient):
    """Test GET /institutions/345/models/10/vers/0."""
    # Authorized.
    response = client.get(
        "/institutions/"
        + uuid_to_str(USER_VALID_INST_UUID)
        + "/models/sample_model_for_school_1/vers/0"
    )
    assert response.status_code == 200
    assert same_model_orderless(response.json(), MODEL_OBJ)


def test_read_inst_model_outputs(client: TestClient):
    """Test GET /institutions/345/models/10/vers/0/output."""
    # Authorized.
    response = client.get(
        "/institutions/"
        + uuid_to_str(USER_VALID_INST_UUID)
        + "/models/sample_model_for_school_1/vers/0/runs"
    )
    assert response.status_code == 200
    assert response.json() == []


def test_read_inst_model_output(client: TestClient):
    """Test GET /institutions/345/models/10/vers/0/output/1."""
    # Authorized.
    response = client.get(
        "/institutions/"
        + uuid_to_str(USER_VALID_INST_UUID)
        + "/models/sample_model_for_school_1/vers/0/run/1"
    )
    assert response.status_code == 404


def test_create_model(client: TestClient):
    """Depending on timeline, fellows may not get to this."""
    response = client.post(
        "/institutions/" + uuid_to_str(USER_VALID_INST_UUID) + "/models/",
        json={"name": "my_model", "schema_configs": "tmp"},
    )

    assert response.status_code == 200


# Retrain a new model.
def test_retrain_model(client: TestClient):
    """Depending on timeline, fellows may not get to this."""
    response = client.post(
        "/institutions/" + uuid_to_str(USER_VALID_INST_UUID) + "/models/123/vers/"
    )
    assert response.status_code == 200
