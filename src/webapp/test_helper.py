"""Helper objects for unit tests across the various files.
"""

from .utilities import AccessType, BaseUser, uuid_to_str
import uuid
from datetime import datetime

DATETIME_TESTING = datetime(2024, 12, 24, 20, 22, 20, 132022)

USER_VALID_INST_UUID = uuid.UUID("1d7c75c3-3eda-4294-9c66-75ea8af97b55")
USER_UUID = uuid.UUID("5301a352-c03d-4a39-beec-16c5668c4700")
USER_1_UUID = uuid.UUID("27316b89-5e04-474a-9ea4-97beaf72c9af")
UUID_INVALID = uuid.UUID("64dbce41-111b-46fe-8e84-c38757477ef2")
SAMPLE_UUID = uuid.UUID("e4862c62-8294-40d8-ab4c-9c298f02f619")

USR = BaseUser(
    uuid_to_str(USER_UUID),
    uuid_to_str(USER_VALID_INST_UUID),
    AccessType.MODEL_OWNER,
    "abc@example.com",
)

VIEWER = BaseUser(
    uuid_to_str(USER_1_UUID),
    uuid_to_str(USER_VALID_INST_UUID),
    AccessType.VIEWER,
    "janesmith@example.com",
)

DATAKINDER = BaseUser(
    uuid_to_str(USER_UUID),
    uuid_to_str(USER_VALID_INST_UUID),
    AccessType.DATAKINDER,
    "taylor@example.com",
)

UNASSIGNED_USER = BaseUser(
    uuid_to_str(SAMPLE_UUID),
    None,
    None,
    "jamie@example.com",
)

BATCH_REQUEST = {
    "name": "batch_foobar",
    "description": "",
    "batch_disabled": False,
}

USER_ACCT_REQUEST = {
    "name": "Taylor Smith",
    "access_type": "DATAKINDER",
    "email": "abc@example.com",
}

USER_ACCT = {
    "name": "Taylor Smith",
    "access_type": "DATAKINDER",
    "email": "abc@example.com",
    "inst_id": uuid_to_str(USER_VALID_INST_UUID),
    "user_id": uuid_to_str(USER_UUID),
}

INSTITUTION_REQ = {
    "name": "foobar school",
    "description": "description of school",
    "state": "NY",
    "retention_days": 1,
    "pdp_id": 12345,
    "is_pdp": True,
    "allowed_schemas": ["UNKNOWN"],
    "allowed_emails": {"foo@foobar.edu": "VIEWER"},
}

INSTITUTION_REQ_BAREBONES = {
    "name": "testing school",
}

EMPTY_INSTITUTION_OBJ = {
    "inst_id": "",
    "name": "",
    "description": "",
    "state": "",
    "pdp_id": None,
    "retention_days": 0,
}

INSTITUTION_OBJ = {
    "inst_id": uuid_to_str(USER_VALID_INST_UUID),
    "name": "valid_school",
    "state": "NY",
    "pdp_id": 12345,
    "description": None,
    "retention_days": None,
}

MODEL_OBJ = {
    "deleted": None,
    "description": None,
    "inst_id": "1d7c75c33eda42949c6675ea8af97b55",
    "m_id": "e4862c62829440d8ab4c9c298f02f619",
    "name": "sample_model_for_school_1",
    "valid": True,
    "vers_id": 0,
}

DATA_OBJ = {
    "batch_ids": [10],
    "name": "foo-data",
    "record_count": 100,
    "retention_days": None,
    "size": 1,
    "description": "some model for foo",
    "uploader": 123,
    "source": "MANUAL_UPLOAD",
    "data_disabled": False,
    "deleted_at": None,
}
