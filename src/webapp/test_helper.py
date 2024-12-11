"""Helper objects for unit tests across the various files.
"""

from .utilities import AccessType, BaseUser, uuid_to_str
import uuid

USR = BaseUser(
    uuid_to_str(uuid.UUID("5301a352-c03d-4a39-beec-16c5668c4700")),
    uuid_to_str(uuid.UUID("1d7c75c3-3eda-4294-9c66-75ea8af97b55")),
    AccessType.MODEL_OWNER,
)
USR_STR = USR.construct_query_param_string()

VIEWER = BaseUser(
    uuid_to_str(uuid.UUID("5301a352-c03d-4a39-beec-16c5668c4700")),
    uuid_to_str(uuid.UUID("1d7c75c3-3eda-4294-9c66-75ea8af97b55")),
    AccessType.VIEWER,
)
VIEWER_STR = VIEWER.construct_query_param_string()

DATAKINDER = BaseUser(
    uuid_to_str(uuid.UUID("5301a352-c03d-4a39-beec-16c5668c4700")),
    uuid_to_str(uuid.UUID("1d7c75c3-3eda-4294-9c66-75ea8af97b55")),
    AccessType.DATAKINDER,
)
DATAKINDER_STR = DATAKINDER.construct_query_param_string()

BATCH_REQUEST = {
    "name": "",
    "description": "",
    "batch_disabled": False,
}

USER_ACCT_REQUEST = {
    "name": "Taylor Smith",
    "access_type": 2,
    "email": "abc@example.com",
}

USER_ACCT = {
    "name": "Taylor Smith",
    "access_type": 2,
    "account_disabled": False,
    "deletion_request": None,
    "email": "abc@example.com",
    "inst_id": uuid_to_str(uuid.UUID("5301a352-c03d-4a39-beec-16c5668c4700")),
    "user_id": "",
    "username": None,
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
    "inst_id": uuid_to_str(uuid.UUID("1d7c75c3-3eda-4294-9c66-75ea8af97b55")),
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
