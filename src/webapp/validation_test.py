"""Test file for file_validation.py.
"""

import pytest

from .validation import (
    valid_subset_lists,
    detect_file_type,
    get_col_names,
    validate_file,
    SchemaType,
)


def test_get_col_names():
    """Testing getting the column names."""
    cols = get_col_names("src/webapp/test_files/test_upload.csv")
    assert cols == ["foo_col", "bar_col", "baz_col"]


def test_valid_subset_lists():
    """Testing valid subset checking."""
    list_a = [1, 2, 3, 4, 5, 6]
    list_b = [1, 2, 3, 4, 6]
    list_c = [5]
    list_d = [3, 4]
    assert valid_subset_lists(list_a, list_b, list_c)
    # Missing value is not in the optional list.
    assert not valid_subset_lists(list_a, list_b, list_d)
    # Subset has an additional element not found in superset.
    assert not valid_subset_lists(list_b, list_a, list_c)


def test_detect_file_type():
    """Testing schema detection."""
    assert (
        detect_file_type(get_col_names("src/webapp/test_files/financial_sst_pdp.csv"))
        == SchemaType.SST_PDP_FINANCE
    )
    assert (
        detect_file_type(get_col_names("src/webapp/test_files/course_sst_pdp.csv"))
        == SchemaType.SST_PDP_COURSE
    )
    assert (
        detect_file_type(get_col_names("src/webapp/test_files/cohort_sst_pdp.csv"))
        == SchemaType.SST_PDP_COHORT
    )
    assert (
        detect_file_type(get_col_names("src/webapp/test_files/course_pdp.csv"))
        == SchemaType.PDP_COURSE
    )
    assert (
        detect_file_type(get_col_names("src/webapp/test_files/cohort_pdp.csv"))
        == SchemaType.PDP_COHORT
    )
    assert (
        detect_file_type(get_col_names("src/webapp/test_files/test_upload.csv"))
        == SchemaType.UNKNOWN
    )
    with pytest.raises(ValueError) as err:
        detect_file_type(get_col_names("src/webapp/test_files/malformed.csv"))
    assert str(err.value) == "CSV file malformed: Could not determine delimiter"


def test_validate_file():
    """Testing file validation."""
    assert validate_file("src/webapp/test_files/financial_sst_pdp.csv")
    assert validate_file("src/webapp/test_files/course_sst_pdp.csv")
    assert validate_file("src/webapp/test_files/cohort_sst_pdp.csv")
    assert validate_file("src/webapp/test_files/course_pdp.csv")
    assert validate_file("src/webapp/test_files/cohort_pdp.csv")
    with pytest.raises(ValueError) as err:
        validate_file("src/webapp/test_files/test_upload.csv")
    assert str(err.value) == "CSV file schema not recognized"
    with pytest.raises(ValueError) as err:
        validate_file("src/webapp/test_files/malformed.csv")
    assert str(err.value) == "CSV file malformed: Could not determine delimiter"
