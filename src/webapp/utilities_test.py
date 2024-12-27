"""Test file for utilities.py.
"""

import pytest

from fastapi import HTTPException
from .utilities import (
    BaseUser,
    has_access_to_inst_or_err,
    has_full_data_access_or_err,
    AccessType,
    uuid_to_str,
)

from .test_helper import USR, DATAKINDER, VIEWER, UUID_INVALID, USER_VALID_INST_UUID


def test_base_user_class_functions():
    """Run tests on various BaseUser class functions."""
    assert DATAKINDER.is_datakinder()
    assert not USR.is_datakinder()

    assert DATAKINDER.has_access_to_inst(uuid_to_str(USER_VALID_INST_UUID))
    assert USR.has_access_to_inst(uuid_to_str(USER_VALID_INST_UUID))
    assert not USR.has_access_to_inst(uuid_to_str(UUID_INVALID))

    assert DATAKINDER.has_full_data_access()
    assert USR.has_full_data_access()
    assert not VIEWER.has_full_data_access()


def test_has_access_to_inst_or_err():
    """Testing valid check for access to institution."""
    with pytest.raises(HTTPException) as err:
        has_access_to_inst_or_err("456", USR)
    assert err.value.status_code == 401
    assert err.value.detail == "Not authorized to read this institution's resources."


def test_has_full_data_access_or_err():
    """Testing valid check for access to full data."""
    with pytest.raises(HTTPException) as err:
        has_full_data_access_or_err(VIEWER, "models")
    assert err.value.status_code == 401
    assert err.value.detail == "Not authorized to view models for this institution."
