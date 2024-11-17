"""Helper objects for unit tests across the various files.
"""

from .utilities import AccessType, BaseUser

USR = BaseUser(12, 345, AccessType.MODEL_OWNER)
USR_STR = USR.construct_query_param_string()

VIEWER = BaseUser(12, 345, AccessType.VIEWER)
VIEWER_STR = VIEWER.construct_query_param_string()

INSTITUTION_OBJ = {
    "m_id": 10,
    "name": "foo-model",
    "vers_id": 0,
    "description": "some model for foo",
    "creator": 123,
    "model_disabled": False,
    "deletion_request": None,
}

BATCH_DATA_OBJ = {
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
