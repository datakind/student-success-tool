"""Test file for utilities.py.
"""

import pytest

from .utilities import get_sftp_bucket_name


def test_get_sftp_bucket_name():
    """Run tests on various BaseUser class functions."""
    assert get_sftp_bucket_name("LOCAL") == "local_sftp_ingestion"
