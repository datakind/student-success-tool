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

USR = BaseUser(
    uuid_to_str(USER_UUID),
    uuid_to_str(USER_VALID_INST_UUID),
    AccessType.MODEL_OWNER,
    "abc@example.com",
)
USR_STR = USR.construct_query_param_string()

VIEWER = BaseUser(
    uuid_to_str(USER_1_UUID),
    uuid_to_str(USER_VALID_INST_UUID),
    AccessType.VIEWER,
    "janesmith@example.com",
)
VIEWER_STR = VIEWER.construct_query_param_string()

DATAKINDER = BaseUser(
    uuid_to_str(USER_UUID),
    uuid_to_str(USER_VALID_INST_UUID),
    AccessType.DATAKINDER,
    "taylor@example.com",
)
DATAKINDER_STR = DATAKINDER.construct_query_param_string()

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
    "account_disabled": False,
    "deletion_request": None,
    "email": "abc@example.com",
    "inst_id": uuid_to_str(USER_VALID_INST_UUID),
    "user_id": uuid_to_str(USER_UUID),
}

INSTITUTION_REQ = {
    "name": "foobar school",
    "description": "description of school",
    "retention_days": 1,
}

EMPTY_INSTITUTION_OBJ = {
    "inst_id": "",
    "name": "",
    "description": "",
    "retention_days": 0,
}

INSTITUTION_OBJ = {
    "inst_id": uuid_to_str(USER_VALID_INST_UUID),
    "name": "valid_school",
    "description": None,
    "retention_days": None,
}

MODEL_OBJ = {
    "m_id": 10,
    "name": "foo-model",
    "vers_id": 0,
    "description": "some model for foo",
    "creator": 123,
    "disabled": False,
    "deletion_request": None,
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
    "deletion_request": None,
}
