import pandas as pd
import pytest

from student_success_tool.dataio import schemas


@pytest.mark.parametrize(
    ["df", "col", "exp"],
    [
        (
            pd.DataFrame({"col1": ["a", "b", "c"]}, dtype="string"),
            "col1",
            pd.Series(["A", "B", "C"], dtype="string"),
        ),
        (
            pd.DataFrame({"col2": ["aBc", "BcD", "cDe"]}, dtype="string"),
            "col2",
            pd.Series(["ABC", "BCD", "CDE"], dtype="string"),
        ),
        (
            pd.DataFrame({"col2": [" a", "B ", " c "]}, dtype="string"),
            "col2",
            pd.Series(["A", "B", "C"], dtype="string"),
        ),
        (
            pd.DataFrame({"col": []}, dtype="string"),
            "col",
            pd.Series([], dtype="string"),
        ),
    ],
)
def test_strip_upper_string_values(df, col, exp):
    obs = schemas.pdp.raw_cohort._strip_upper_string_values(df, col=col)
    assert obs.equals(exp)


@pytest.mark.parametrize(
    ["df", "col", "to_replace", "exp"],
    [
        (
            pd.DataFrame({"gpa1": ["4.0", "2.5", "UK"]}),
            "gpa1",
            "UK",
            pd.Series(["4.0", "2.5", None]),
        ),
        (
            pd.DataFrame({"col": ["-1.0", "1", "2"]}),
            "col",
            "-1.0",
            pd.Series([None, "1", "2"]),
        ),
    ],
)
def test_replace_values_with_null(df, col, to_replace, exp):
    obs = schemas.pdp.raw_cohort._replace_values_with_null(
        df, col=col, to_replace=to_replace
    )
    assert obs.equals(exp)


@pytest.mark.parametrize(
    ["df", "col", "exp"],
    [
        (
            pd.DataFrame({"col1": ["1", "0", "1"]}, dtype="string"),
            "col1",
            pd.Series([True, False, True], dtype="boolean"),
        ),
        (
            pd.DataFrame({"col2": ["1", "0", None]}, dtype="string"),
            "col2",
            pd.Series([True, False, None], dtype="boolean"),
        ),
        (
            pd.DataFrame({"col3": ["True", "False"]}, dtype="string"),
            "col3",
            pd.Series([True, False], dtype="boolean"),
        ),
    ],
)
def test_cast_to_bool_via_int(df, col, exp):
    obs = schemas.pdp.raw_cohort._cast_to_bool_via_int(df, col=col)
    assert obs.equals(exp)
