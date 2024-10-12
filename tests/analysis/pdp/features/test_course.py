import pandas as pd
import pytest

from student_success_tool.analysis.pdp import constants
from student_success_tool.analysis.pdp.features import course


@pytest.mark.parametrize(
    ["df", "min_passing_grade", "course_level_pattern", "exp"],
    [
        (
            pd.DataFrame(
                {
                    "academic_year": ["2020-21", "2020-21", "2021-22"],
                    "academic_term": ["FALL", "SPRING", "FALL"],
                    "course_prefix": ["MATH", "MATH", "PHYS"],
                    "course_number": ["101", "202", "303"],
                    "course_cip": ["40.02", "45.06", "03.01"],
                    "course_type": ["CU", "CU", "CC"],
                    "delivery_method": ["O", "H", "F"],
                    "grade": ["4", "1", "W"],
                }
            ).astype({"grade": "category"}),
            "2",
            constants.DEFAULT_COURSE_LEVEL_PATTERN,
            pd.DataFrame(
                {
                    "academic_year": ["2020-21", "2020-21", "2021-22"],
                    "academic_term": ["FALL", "SPRING", "FALL"],
                    "course_prefix": ["MATH", "MATH", "PHYS"],
                    "course_number": ["101", "202", "303"],
                    "course_cip": ["40.02", "45.06", "03.01"],
                    "course_type": ["CU", "CU", "CC"],
                    "delivery_method": ["O", "H", "F"],
                    "grade": ["4", "1", "W"],
                    "course_id": ["MATH101", "MATH202", "PHYS303"],
                    "course_subject_area": [40, 45, 3],
                    "course_passed": [True, False, pd.NA],
                    "course_completed": [True, True, False],
                    "course_level": [1, 2, 3],
                    "course_grade_numeric": [4, 1, pd.NA],
                }
            ).astype({"course_passed": "boolean", "course_grade_numeric": "Int8"}),
        ),
    ],
)
def test_add_course_features(df, min_passing_grade, course_level_pattern, exp):
    obs = course.add_features(
        df,
        min_passing_grade=min_passing_grade,
        course_level_pattern=course_level_pattern,
    )
    assert isinstance(obs, pd.DataFrame) and not obs.empty
    assert obs.equals(exp) or obs.compare(exp).empty


@pytest.mark.parametrize(
    ["df", "prefix_col", "number_col", "exp"],
    [
        (
            pd.DataFrame(
                {
                    "course_prefix": ["MATH", "BIO", "PHYS", "X", "Y", "Z"],
                    "course_number": ["101", "202", "303A", "404AB", "99", "001"],
                }
            ),
            "course_prefix",
            "course_number",
            pd.Series(["MATH101", "BIO202", "PHYS303A", "X404AB", "Y99", "Z001"]),
        ),
    ],
)
def test_course_id(df, prefix_col, number_col, exp):
    obs = course.course_id(df, prefix_col=prefix_col, number_col=number_col)
    assert isinstance(obs, pd.Series) and not obs.empty
    assert obs.equals(exp) or obs.compare(exp).empty


@pytest.mark.parametrize(
    ["df", "col", "min_passing_grade", "exp"],
    [
        (
            pd.DataFrame({"grade": ["4", "2", "P", "F", "0"]}).astype(
                {"grade": pd.CategoricalDtype(["0", "1", "2", "3", "4", "P", "F"])}
            ),
            "grade",
            "1",
            pd.Series([True, True, True, False, False], dtype="boolean"),
        ),
        (
            pd.DataFrame({"grade": ["4", "1", "P", "F", "0", "I", "W"]}).astype(
                {
                    "grade": pd.CategoricalDtype(
                        ["0", "1", "2", "3", "4", "P", "F", "W", "I"]
                    )
                }
            ),
            "grade",
            "2",
            pd.Series([True, False, True, False, False, pd.NA, pd.NA], dtype="boolean"),
        ),
    ],
)
def test_course_passed(df, col, min_passing_grade, exp):
    obs = course.course_passed(df, col=col, min_passing_grade=min_passing_grade)
    assert isinstance(obs, pd.Series) and not obs.empty
    assert obs.equals(exp) or obs.compare(exp).empty


@pytest.mark.parametrize(
    ["df", "col", "exp"],
    [
        (
            pd.DataFrame({"grade": ["4", "1", "P", "F", "0", "I", "W"]}).astype(
                {
                    "grade": pd.CategoricalDtype(
                        ["0", "1", "2", "3", "4", "P", "F", "W", "I"]
                    )
                }
            ),
            "grade",
            pd.Series([True, True, True, True, True, False, False]),
        ),
    ],
)
def test_course_completed(df, col, exp):
    obs = course.course_completed(df, col=col)
    assert isinstance(obs, pd.Series) and not obs.empty
    assert obs.equals(exp) or obs.compare(exp).empty


@pytest.mark.parametrize(
    ["df", "col", "pattern", "exp"],
    [
        (
            pd.DataFrame(
                {"course_number": ["101", "202", "303A", "404AB", "404ABC", "99", "1"]}
            ),
            "course_number",
            r"^(?P<course_level>\d)\d{2}(?:[A-Z]{,2})?$",
            pd.Series([1, 2, 3, 4, pd.NA, pd.NA, pd.NA], dtype="Int32"),
        ),
    ],
)
def test_course_level(df, col, pattern, exp):
    obs = course.course_level(df, col=col, pattern=pattern)
    assert isinstance(obs, pd.Series) and not obs.empty
    assert obs.equals(exp) or obs.compare(exp).empty


@pytest.mark.parametrize(
    ["df", "col", "exp"],
    [
        (
            pd.DataFrame({"grade": ["4", "2", "P", "F", "0"]}).astype(
                {"grade": pd.CategoricalDtype(["0", "1", "2", "3", "4", "P", "F"])}
            ),
            "grade",
            pd.Series(["4", "2", pd.NA, pd.NA, "0"], dtype="Int8"),
        ),
    ],
)
def test_course_grade_numeric(df, col, exp):
    obs = course.course_grade_numeric(df, col=col)
    assert isinstance(obs, pd.Series) and not obs.empty
    assert obs.equals(exp) or obs.compare(exp).empty
