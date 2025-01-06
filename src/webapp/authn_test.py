"""Test file for authn.py.
"""

import pytest

from fastapi import HTTPException
import uuid
from .authn import (
    get_password_hash,
    verify_password,
)

PASSWORD_STR = "pass123"


def test_password_functions():
    """Run tests on various password functions."""
    pass_hash = get_password_hash(PASSWORD_STR)
    assert pass_hash[0:7] == "$2y$12$"
    assert verify_password(PASSWORD_STR, pass_hash)
