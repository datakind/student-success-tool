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
    databricksify_inst_name,
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


def test_databricksify_inst_name():
    """
    Testing databricksifying institution name
    """
    assert (
        databricksify_inst_name("Motlow State Community College") == "motlow_state_cc"
    )
    assert (
        databricksify_inst_name("Metro State University Denver")
        == "metro_state_uni_denver"
    )
    assert databricksify_inst_name("Kentucky State University") == "kentucky_state_uni"
    assert databricksify_inst_name("Central Arizona College") == "central_arizona_col"
    assert (
        databricksify_inst_name("Harrisburg University of Science and Technology")
        == "harrisburg_uni_st"
    )
    assert (
        databricksify_inst_name("Southeast Kentucky community technical college")
        == "southeast_kentucky_ctc"
    )
    assert (
        databricksify_inst_name("Northwest State Community College")
        == "northwest_state_cc"
    )

    with pytest.raises(ValueError) as err:
        databricksify_inst_name("Northwest (invalid)")
    assert str(err.value) == "Unexpected character found in Databricks compatible name."
