"""Test file for utilities.py.
"""

import pytest

from fastapi import HTTPException
from .utilities import (
    BaseUser,
    has_access_to_inst_or_err,
    has_full_data_access_or_err,
    AccessType,
)

test_datakinder = BaseUser(1, 123, AccessType.DATAKINDER)
test_model_owner = BaseUser(2, 123, AccessType.MODEL_OWNER)
test_data_owner = BaseUser(3, 123, AccessType.DATA_OWNER)
test_viewer = BaseUser(4, 123, AccessType.VIEWER)
test_api = BaseUser(None, 123, AccessType.MODEL_OWNER)


def test_base_user_class_functions():
    """Run tests on various BaseUser class functions."""
    assert test_datakinder.is_datakinder()
    assert not test_model_owner.is_datakinder()
    assert not test_api.is_datakinder()

    assert test_datakinder.has_access_to_inst(567)
    assert not test_data_owner.has_access_to_inst(567)
    assert test_model_owner.has_access_to_inst(123)

    assert test_datakinder.has_full_data_access()
    assert test_data_owner.has_full_data_access()
    assert test_model_owner.has_full_data_access()
    assert not test_viewer.has_full_data_access()


def test_has_access_to_inst_or_err():
    """Testing valid check for access to institution."""
    with pytest.raises(HTTPException) as err:
        has_access_to_inst_or_err(456, test_model_owner)
    assert err.value.status_code == 401
    assert err.value.detail == "Not authorized to read this institution's resources."


def test_has_full_data_access_or_err():
    """Testing valid check for access to full data."""
    with pytest.raises(HTTPException) as err:
        has_full_data_access_or_err(test_viewer, "models")
    assert err.value.status_code == 401
    assert err.value.detail == "Not authorized to view models for this institution."
