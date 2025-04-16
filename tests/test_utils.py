from collections.abc import Iterable

import pytest

from student_success_tool import utils


@pytest.mark.parametrize(
    ["val", "exp"],
    [
        ([1, 2, 3], [1, 2, 3]),
        ((1, 2, 3), [1, 2, 3]),
        ({"a": 1, "b": 2}, ["a", "b"]),
        ("abc", ["abc"]),
        (1, [1]),
        (None, [None]),
    ],
)
def test_to_list(val, exp):
    obs = utils.types.to_list(val)
    assert obs == exp


@pytest.mark.parametrize(
    ["value", "exp"],
    [
        ([1, 2, 3], True),
        ({"a", "b", "c"}, True),
        ((True, False, True), True),
        ("string", False),
        (b"bytes", False),
    ],
)
def test_is_collection_but_not_string(value, exp):
    obs = utils.types.is_collection_but_not_string(value)
    assert isinstance(obs, bool)
    assert obs == exp


@pytest.mark.parametrize(
    ["eles", "exp"],
    [
        ([2, 1, 2, 2, 1, 3], [2, 1, 3]),
        (("a", "c", "b", "b", "c", "a"), ["a", "c", "b"]),
    ],
)
def test_unique_elements_in_order(eles, exp):
    obs = utils.misc.unique_elements_in_order(eles)
    assert isinstance(obs, Iterable)
    assert list(obs) == exp


@pytest.mark.parametrize(
    ["val", "exp"],
    [
        ("Student GUID", "student_guid"),
        ("Credential Type Sought Year 1", "credential_type_sought_year_1"),
        ("Years to Bachelors at cohort inst.", "years_to_bachelors_at_cohort_inst"),
        ("Enrolled at Other Institution(s)", "enrolled_at_other_institution_s"),
    ],
)
def test_convert_to_snake_case(val, exp):
    obs = utils.misc.convert_to_snake_case(val)
    assert obs == exp
