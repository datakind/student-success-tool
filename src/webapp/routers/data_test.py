"""Test file for the data.py file and constituent API functions.
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
)
from ..main import app
from ..database import (
    FileTable,
    BatchTable,
    InstTable,
    Base,
    get_session,
    local_session,
)
from ..utilities import uuid_to_str, get_current_active_user, SchemaType
from .data import router, DataOverview, DataInfo, BatchCreationRequest
from collections import Counter
from ..gcsutil import StorageControl

MOCK_STORAGE = mock.Mock()

UUID_2 = uuid.UUID("9bcbc782-2e71-4441-afa2-7a311024a5ec")
FILE_UUID_1 = uuid.UUID("f0bb3a20-6d92-4254-afed-6a72f43c562a")
FILE_UUID_2 = uuid.UUID("cb02d06c-2a59-486a-9bdd-d394a4fcb833")
FILE_UUID_3 = uuid.UUID("fbe67a2e-50e0-40c7-b7b8-07043cb813a5")
BATCH_UUID = uuid.UUID("5b2420f3-1035-46ab-90eb-74d5df97de43")
CREATOR_UUID = uuid.UUID("0ad8b77c-49fb-459a-84b1-8d2c05722c4a")


# Orderless comparison of two iterables.
def counter_repr(x):
    return {frozenset(Counter(item).items()) for item in x}


# Compares two DataOverview objects.
def same_file_orderless(a_elem: DataInfo, b_elem: DataInfo):
    if (
        a_elem["inst_id"] != b_elem["inst_id"]
        or counter_repr(a_elem["batch_ids"]) != counter_repr(b_elem["batch_ids"])
        or a_elem["name"] != b_elem["name"]
        or a_elem["description"] != b_elem["description"]
        or a_elem["uploader"] != b_elem["uploader"]
        or a_elem["deleted"] != b_elem["deleted"]
        or a_elem["source"] != b_elem["source"]
        or a_elem["deletion_request_time"] != b_elem["deletion_request_time"]
        or a_elem["retention_days"] != b_elem["retention_days"]
        or a_elem["sst_generated"] != b_elem["sst_generated"]
        or a_elem["valid"] != b_elem["valid"]
        or a_elem["uploaded_date"] != b_elem["uploaded_date"]
    ):
        return False
    return True


# Compares two DataOverview objects.
def same_orderless(a: DataOverview, b: DataOverview):
    for a_elem in a["batches"]:
        found = False
        for b_elem in b["batches"]:
            if a_elem["batch_id"] != b_elem["batch_id"]:
                continue
            found = True
            if (
                a_elem["inst_id"] != b_elem["inst_id"]
                or counter_repr(a_elem["file_ids"]) != counter_repr(b_elem["file_ids"])
                or a_elem["name"] != b_elem["name"]
                or a_elem["description"] != b_elem["description"]
                or a_elem["creator"] != b_elem["creator"]
                or a_elem["deleted"] != b_elem["deleted"]
                or a_elem["completed"] != b_elem["completed"]
                or a_elem["deletion_request_time"] != b_elem["deletion_request_time"]
                or a_elem["created_date"] != b_elem["created_date"]
            ):
                return False
        if not found:
            return False
    for a_elem in a["files"]:
        found = False
        for b_elem in b["files"]:
            if a_elem["data_id"] != b_elem["data_id"]:
                continue
            found = True
            if not same_file_orderless(a_elem, b_elem):
                return False
        if not found:
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
        creator=CREATOR_UUID,
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


def test_read_inst_all_input_files(client: TestClient):
    """Test GET /institutions/<uuid>/input."""
    response = client.get("/institutions/" + uuid_to_str(UUID_INVALID) + "/input")

    assert str(response) == "<Response [401 Unauthorized]>"
    assert (
        response.text
        == '{"detail":"Not authorized to read this institution\'s resources."}'
    )
    # Authorized.
    response = client.get(
        "/institutions/" + uuid_to_str(USER_VALID_INST_UUID) + "/input"
    )
    assert response.status_code == 200
    assert same_orderless(
        response.json(),
        {
            "batches": [
                {
                    "batch_id": "5b2420f3103546ab90eb74d5df97de43",
                    "inst_id": "1d7c75c33eda42949c6675ea8af97b55",
                    "file_ids": [
                        "f0bb3a206d924254afed6a72f43c562a",
                        "fbe67a2e50e040c7b7b807043cb813a5",
                    ],
                    "name": "batch_foo",
                    "description": None,
                    "creator": "0ad8b77c49fb459a84b18d2c05722c4a",
                    "deleted": False,
                    "completed": False,
                    "deletion_request_time": None,
                    "created_date": "2024-12-24T20:22:20.132022",
                }
            ],
            "files": [
                {
                    "name": "file_input_one",
                    "data_id": "f0bb3a206d924254afed6a72f43c562a",
                    "batch_ids": ["5b2420f3103546ab90eb74d5df97de43"],
                    "inst_id": "1d7c75c33eda42949c6675ea8af97b55",
                    "description": None,
                    "uploader": "",
                    "source": "MANUAL_UPLOAD",
                    "deleted": False,
                    "deletion_request_time": None,
                    "retention_days": None,
                    "sst_generated": False,
                    "valid": True,
                    "uploaded_date": "2024-12-24T20:22:20.132022",
                },
                {
                    "name": "file_input_two",
                    "data_id": "cb02d06c2a59486a9bddd394a4fcb833",
                    "batch_ids": [],
                    "inst_id": "1d7c75c33eda42949c6675ea8af97b55",
                    "description": None,
                    "uploader": "",
                    "source": "PDP_SFTP",
                    "deleted": False,
                    "deletion_request_time": None,
                    "retention_days": None,
                    "sst_generated": False,
                    "valid": False,
                    "uploaded_date": "2024-12-24T20:22:20.132022",
                },
            ],
        },
    )


def test_read_inst_all_output_files(client: TestClient):
    """Test GET /institutions/<uuid>/output."""
    response = client.get("/institutions/" + uuid_to_str(UUID_INVALID) + "/output")

    assert str(response) == "<Response [401 Unauthorized]>"
    assert (
        response.text
        == '{"detail":"Not authorized to read this institution\'s resources."}'
    )
    # Authorized.
    response = client.get(
        "/institutions/" + uuid_to_str(USER_VALID_INST_UUID) + "/output"
    )
    assert response.status_code == 200
    assert same_orderless(
        response.json(),
        {
            "batches": [
                {
                    "batch_id": "5b2420f3103546ab90eb74d5df97de43",
                    "inst_id": "1d7c75c33eda42949c6675ea8af97b55",
                    "file_ids": [
                        "fbe67a2e50e040c7b7b807043cb813a5",
                        "f0bb3a206d924254afed6a72f43c562a",
                    ],
                    "name": "batch_foo",
                    "description": None,
                    "creator": "0ad8b77c49fb459a84b18d2c05722c4a",
                    "deleted": False,
                    "completed": False,
                    "deletion_request_time": None,
                    "created_date": "2024-12-24T20:22:20.132022",
                }
            ],
            "files": [
                {
                    "name": "file_output_one",
                    "data_id": "fbe67a2e50e040c7b7b807043cb813a5",
                    "batch_ids": ["5b2420f3103546ab90eb74d5df97de43"],
                    "inst_id": "1d7c75c33eda42949c6675ea8af97b55",
                    "description": None,
                    "uploader": "",
                    "source": None,
                    "deleted": False,
                    "deletion_request_time": None,
                    "retention_days": None,
                    "sst_generated": True,
                    "valid": True,
                    "uploaded_date": "2024-12-24T20:22:20.132022",
                }
            ],
        },
    )


def test_read_batch_info(client: TestClient):
    """Test GET /institutions/<uuid>/batch/<uuid>."""
    response = client.get(
        "/institutions/"
        + uuid_to_str(UUID_INVALID)
        + "/batch/"
        + uuid_to_str(BATCH_UUID)
    )

    assert str(response) == "<Response [401 Unauthorized]>"
    assert (
        response.text
        == '{"detail":"Not authorized to read this institution\'s resources."}'
    )
    # Authorized.
    response = client.get(
        "/institutions/"
        + uuid_to_str(USER_VALID_INST_UUID)
        + "/batch/"
        + uuid_to_str(BATCH_UUID)
    )
    assert response.status_code == 200
    assert same_orderless(
        response.json(),
        {
            "batches": [
                {
                    "batch_id": "5b2420f3103546ab90eb74d5df97de43",
                    "inst_id": "1d7c75c33eda42949c6675ea8af97b55",
                    "file_ids": [
                        "f0bb3a206d924254afed6a72f43c562a",
                        "fbe67a2e50e040c7b7b807043cb813a5",
                    ],
                    "name": "batch_foo",
                    "description": None,
                    "creator": "0ad8b77c49fb459a84b18d2c05722c4a",
                    "deleted": False,
                    "completed": False,
                    "deletion_request_time": None,
                    "created_date": "2024-12-24T20:22:20.132022",
                }
            ],
            "files": [
                {
                    "name": "file_output_one",
                    "data_id": "fbe67a2e50e040c7b7b807043cb813a5",
                    "batch_ids": ["5b2420f3103546ab90eb74d5df97de43"],
                    "inst_id": "1d7c75c33eda42949c6675ea8af97b55",
                    "description": None,
                    "uploader": "",
                    "source": None,
                    "deleted": False,
                    "deletion_request_time": None,
                    "retention_days": None,
                    "sst_generated": True,
                    "valid": True,
                    "uploaded_date": "2024-12-24T20:22:20.132022",
                },
                {
                    "name": "file_input_one",
                    "data_id": "f0bb3a206d924254afed6a72f43c562a",
                    "batch_ids": ["5b2420f3103546ab90eb74d5df97de43"],
                    "inst_id": "1d7c75c33eda42949c6675ea8af97b55",
                    "description": None,
                    "uploader": "",
                    "source": "MANUAL_UPLOAD",
                    "deleted": False,
                    "deletion_request_time": None,
                    "retention_days": None,
                    "sst_generated": False,
                    "valid": True,
                    "uploaded_date": "2024-12-24T20:22:20.132022",
                },
            ],
        },
    )


def test_read_file_id_info(client: TestClient):
    """Test GET /institutions/<uuid>/file/<uuid>."""
    response = client.get(
        "/institutions/"
        + uuid_to_str(UUID_INVALID)
        + "/file_id/"
        + uuid_to_str(FILE_UUID_1)
    )

    assert str(response) == "<Response [401 Unauthorized]>"
    assert (
        response.text
        == '{"detail":"Not authorized to read this institution\'s resources."}'
    )
    # Authorized.
    response = client.get(
        "/institutions/"
        + uuid_to_str(USER_VALID_INST_UUID)
        + "/file_id/"
        + uuid_to_str(FILE_UUID_1)
    )
    assert response.status_code == 200
    assert same_file_orderless(
        response.json(),
        {
            "name": "file_input_one",
            "data_id": "f0bb3a206d924254afed6a72f43c562a",
            "batch_ids": ["5b2420f3103546ab90eb74d5df97de43"],
            "inst_id": "1d7c75c33eda42949c6675ea8af97b55",
            "description": None,
            "uploader": "",
            "source": "MANUAL_UPLOAD",
            "deleted": False,
            "deletion_request_time": None,
            "retention_days": None,
            "sst_generated": False,
            "valid": True,
            "uploaded_date": "2024-12-24T20:22:20.132022",
        },
    )


def test_create_batch(client: TestClient):
    """Test POST /institutions/<uuid>/batch."""
    response = client.post(
        "/institutions/" + uuid_to_str(UUID_INVALID) + "/batch",
        json={"name": "batch_name_foo"},
    )
    assert str(response) == "<Response [401 Unauthorized]>"
    assert (
        response.text
        == '{"detail":"Not authorized to read this institution\'s resources."}'
    )
    # Authorized.
    response = client.post(
        "/institutions/" + uuid_to_str(USER_VALID_INST_UUID) + "/batch",
        json={
            "name": "batch_foobar",
            "description": "",
            "batch_disabled": "False",
            "file_ids": [uuid_to_str(FILE_UUID_1)],
            "file_names": ["file_input_one", "file_input_two"],
        },
    )
    assert response.status_code == 200
    assert response.json()["name"] == "batch_foobar"
    assert response.json()["description"] == ""
    assert response.json()["creator"] == uuid_to_str(USER_UUID)
    assert response.json()["deleted"] == False
    assert response.json()["completed"] == False
    assert response.json()["deletion_request_time"] == None
    assert response.json()["inst_id"] == uuid_to_str(USER_VALID_INST_UUID)
    assert uuid_to_str(FILE_UUID_2) in response.json()["file_ids"]
    assert uuid_to_str(FILE_UUID_1) in response.json()["file_ids"]
    assert len(response.json()["file_ids"]) == 2


def test_update_batch(client: TestClient):
    """Test PATCH /institutions/<uuid>/batch."""
    response = client.patch(
        "/institutions/"
        + uuid_to_str(UUID_INVALID)
        + "/batch/"
        + uuid_to_str(BATCH_UUID),
        json={"name": "batch_name_updated_foo"},
    )
    assert str(response) == "<Response [401 Unauthorized]>"
    assert (
        response.text
        == '{"detail":"Not authorized to read this institution\'s resources."}'
    )
    # Authorized.
    response = client.patch(
        "/institutions/"
        + uuid_to_str(USER_VALID_INST_UUID)
        + "/batch/"
        + uuid_to_str(BATCH_UUID),
        json={
            "name": "batch_name_updated_foo",
            "completed": "True",
            "file_ids": [uuid_to_str(FILE_UUID_2)],
        },
    )
    assert response.status_code == 200
    assert response.json()["name"] == "batch_name_updated_foo"
    assert response.json()["description"] == None
    assert response.json()["creator"] == uuid_to_str(CREATOR_UUID)
    assert response.json()["deleted"] == None
    assert response.json()["completed"] == True
    assert response.json()["deletion_request_time"] == None
    assert response.json()["inst_id"] == uuid_to_str(USER_VALID_INST_UUID)
    assert response.json()["file_ids"] == [uuid_to_str(FILE_UUID_2)]


def test_validate_success_batch(client: TestClient):
    """Test PATCH /institutions/<uuid>/batch."""
    MOCK_STORAGE.validate_file.return_value = {SchemaType.UNKNOWN}
    response = client.post(
        "/institutions/" + uuid_to_str(UUID_INVALID) + "/input/validate/file_name.csv",
    )
    assert str(response) == "<Response [401 Unauthorized]>"
    assert (
        response.text
        == '{"detail":"Not authorized to read this institution\'s resources."}'
    )
    # Authorized.
    response = client.post(
        "/institutions/"
        + uuid_to_str(USER_VALID_INST_UUID)
        + "/input/validate/file_name.csv",
    )
    assert response.status_code == 200
    assert response.json()["name"] == "file_name.csv"
    assert response.json()["file_types"] == ["UNKNOWN"]
    assert response.json()["inst_id"] == uuid_to_str(USER_VALID_INST_UUID)


def test_validate_failure_batch(client: TestClient):
    """Test PATCH /institutions/<uuid>/batch."""
    MOCK_STORAGE.validate_file.return_value = {SchemaType.PDP_COHORT}
    # Authorized.
    response = client.post(
        "/institutions/"
        + uuid_to_str(USER_VALID_INST_UUID)
        + "/input/validate/file_name.csv",
    )
    assert response.status_code == 200
    assert response.json()["name"] == "file_name.csv"
    assert response.json()["file_types"] == ["PDP_COHORT"]
    assert response.json()["inst_id"] == uuid_to_str(USER_VALID_INST_UUID)


def test_pull_pdp_sftp(client: TestClient):
    """Test POST /institutions/345/input/pdp_sftp/."""
    # Authorized.
    response = client.post("/institutions/345/input/pdp_sftp")
    # assert response.status_code == 200


def test_upload_file(client: TestClient):
    """Test POST /institutions/345/input/."""
    # Authorized.
    response = client.post("/institutions/345/input")
    # assert response.status_code == 200


def test_download_inst_file(client: TestClient):
    """Test GET /institutions/<uuid>/output/<uuid>."""
    # Authorized.
    response = client.get("/institutions/345/output/10")
    # assert response.status_code == 200
