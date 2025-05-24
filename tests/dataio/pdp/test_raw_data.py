import os
from contextlib import nullcontext as does_not_raise

import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from student_success_tool import dataio

FIXTURES_PATH = "tests/fixtures"


@pytest.mark.parametrize(
    ["file_name", "schema", "kwargs"],
    [
        ("raw_pdp_course_data.csv", None, None),
        ("raw_pdp_course_data.csv", dataio.schemas.pdp.RawPDPCourseDataSchema, None),
        (
            "raw_pdp_course_data.csv",
            dataio.schemas.pdp.RawPDPCourseDataSchema,
            {"nrows": 1},
        ),
    ],
)
def test_read_raw_pdp_course_data(file_name, schema, kwargs):
    file_path = os.path.join(FIXTURES_PATH, file_name)
    result = dataio.pdp.read_raw_course_data(
        file_path=file_path, schema=schema, dttm_format="%Y%m%d", **(kwargs or {})
    )
    assert isinstance(result, pd.DataFrame)
    assert not result.empty


@pytest.mark.parametrize(
    ["file_name", "schema", "converter_func", "exp_ctx"],
    [
        (
            "raw_pdp_course_data_invalid.csv",
            dataio.schemas.pdp.RawPDPCourseDataSchema,
            None,
            pytest.raises(SchemaErrors),
        ),
        (
            "raw_pdp_course_data_invalid.csv",
            dataio.schemas.pdp.RawPDPCourseDataSchema,
            lambda df: df.drop_duplicates(subset=["institution_id", "student_guid"]),
            does_not_raise(),
        ),
    ],
)
def test_read_raw_pdp_course_data_convert(file_name, schema, converter_func, exp_ctx):
    file_path = os.path.join(FIXTURES_PATH, file_name)
    with exp_ctx:
        result = dataio.pdp.read_raw_course_data(
            file_path=file_path,
            schema=schema,
            dttm_format="%Y%m%d",
            converter_func=converter_func,
        )
        assert isinstance(result, pd.DataFrame)
        assert not result.empty


@pytest.mark.parametrize(
    ["file_name", "schema", "kwargs"],
    [
        ("raw_pdp_cohort_data.csv", None, None),
        ("raw_pdp_cohort_data.csv", dataio.schemas.pdp.RawPDPCohortDataSchema, None),
        # Yes and No replace 1 and 0.
        ("raw_pdp_cohort_data_ys.csv", dataio.schemas.pdp.RawPDPCohortDataSchema, None),
        (
            "raw_pdp_cohort_data.csv",
            dataio.schemas.pdp.RawPDPCohortDataSchema,
            {"nrows": 1},
        ),
    ],
)
def test_read_raw_pdp_cohort_data(file_name, schema, kwargs):
    file_path = os.path.join(FIXTURES_PATH, file_name)
    result = dataio.pdp.read_raw_cohort_data(
        file_path=file_path, schema=schema, **(kwargs or {})
    )
    assert isinstance(result, pd.DataFrame)
    assert not result.empty


@pytest.mark.parametrize(
    ["df", "col", "fmt", "exp"],
    [
        (
            pd.DataFrame({"dttm": ["2024-01-01", "2024-01-02"]}, dtype="string"),
            "dttm",
            "%Y-%m-%d",
            pd.Series([pd.Timestamp(2024, 1, 1), pd.Timestamp(2024, 1, 2)]),
        ),
        (
            pd.DataFrame({"foo": ["20240101.0", "20240102.0"]}, dtype="string"),
            "foo",
            "%Y%m%d.0",
            pd.Series([pd.Timestamp(2024, 1, 1), pd.Timestamp(2024, 1, 2)]),
        ),
    ],
)
def test_parse_dttm_values(df, col, fmt, exp):
    obs = dataio.pdp.raw_data._parse_dttm_values(df, col=col, fmt=fmt)
    assert obs.equals(exp)
