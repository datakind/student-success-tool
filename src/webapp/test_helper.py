"""Helper objects for unit tests across the various files.
"""

from .utilities import AccessType, BaseUser

USR = BaseUser(12, 345, AccessType.MODEL_OWNER)
USR_STR = USR.construct_query_param_string()

VIEWER = BaseUser(12, 345, AccessType.VIEWER)
VIEWER_STR = VIEWER.construct_query_param_string()

DATAKINDER = BaseUser(12, 345, AccessType.DATAKINDER)
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
    "inst_id": 345,
    "user_id": 1,
    "username": None,
}

INSTITUTION_OBJ = {
    "inst_id": 345,
    "name": "foobar school",
    "description": "School foobar",
    "retention_days": 1,
}

EMPTY_INSTITUTION_OBJ = {
    "inst_id": 345,
    "name": "",
    "description": "",
    "retention_days": 0,
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
    "batch_id": 10,
    "name": "foo-data",
    "record_count": 100,
    "size": 1,
    "description": "some model for foo",
    "uploader": 123,
    "source": "MANUAL_UPLOAD",
    "data_disabled": False,
    "deletion_request": None,
}
