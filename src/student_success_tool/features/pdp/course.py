import functools as ft
import logging

import pandas as pd

from . import constants, shared

LOGGER = logging.getLogger(__name__)

NON_NUMERIC_GRADES = {"A", "F", "I", "M", "O", "P", "W"}
NON_PASS_FAIL_GRADES = {"A", "I", "M", "O", "W"}
NON_COMPLETE_GRADES = {"I", "W"}


def add_features(
    df: pd.DataFrame,
    *,
    min_passing_grade: float = constants.DEFAULT_MIN_PASSING_GRADE,
    course_level_pattern: str = constants.DEFAULT_COURSE_LEVEL_PATTERN,
) -> pd.DataFrame:
    """
    Compute course-level features from a pdp course dataset,
    and add as columns to ``df`` .

    Args:
        df
        min_passing_grade: Minimum numeric grade considered by institution as "passing".
            Note that this is represented as a float, while grades are strings
            since the values include both numeric and alpha-categorical values.
            This value is only compared against numeric grades; relevant categoricals
            are handled appropriately, e.g. "P" => "Pass" is always considered "passing".
        course_level_pattern: Regex string that extracts a course's level from its number
            (e.g. 1 from "101"). *Must* include exactly one capture group,
            which is taken to be the course level.
    """
    LOGGER.info("adding course features ...")
    return df.assign(
        course_id=course_id,
        course_subject_area=course_subject_area,
        course_passed=ft.partial(course_passed, min_passing_grade=min_passing_grade),
        course_completed=course_completed,
        course_level=ft.partial(course_level, pattern=course_level_pattern),
        course_grade_numeric=course_grade_numeric,
        course_grade=course_grade,
    )


def course_id(
    df: pd.DataFrame,
    *,
    prefix_col: str = "course_prefix",
    number_col: str = "course_number",
) -> pd.Series:
    return df[prefix_col].str.cat(df[number_col], sep="")


def course_subject_area(df: pd.DataFrame, *, col: str = "course_cip") -> pd.Series:
    return shared.extract_short_cip_code(df[col])


def course_passed(
    df: pd.DataFrame, *, col: str = "grade", min_passing_grade: float
) -> pd.Series:
    series = (
        df[col]
        .astype("string")
        .map(
            ft.partial(_grade_is_passing, min_passing_grade=min_passing_grade),  # type: ignore
            na_action="ignore",
        )
        .astype("boolean")
    )
    assert isinstance(series, pd.Series)  # type guard
    return series


def course_completed(df: pd.DataFrame, *, col: str = "grade") -> pd.Series:
    return ~(df[col].astype("string").isin(NON_COMPLETE_GRADES))


def course_level(
    df: pd.DataFrame, *, col: str = "course_number", pattern: str
) -> pd.Series:
    return (
        df[col]
        .astype("string")
        .str.strip()
        .str.extract(pattern, expand=False)
        .astype("Int8")
    )


def course_grade_numeric(df: pd.DataFrame, *, col: str = "grade") -> pd.Series:
    return df[col].mask(df[col].isin(NON_NUMERIC_GRADES), pd.NA).astype("Float32")


def course_grade(
    df: pd.DataFrame,
    *,
    grade_col: str = "grade",
    grade_num_col: str = "course_grade_numeric",
) -> pd.Series:
    non_numeric_grades = (
        df[grade_col]
        .mask(~df[grade_col].isin(NON_NUMERIC_GRADES), pd.NA)
        # frustratingly, pdp uses "A" grade to indicate "Audit", which is just begging
        # for confusion with the usual meaning of an "A" grade :/
        # let's replace it with "AUDIT" for clarity, and so we can safely combine
        # non-numeric grades with derived letter grades below
        .replace("A", value="AUDIT")
        # similarly, "O" looks like "0", so let's replace with "OTHER" for clarity
        .replace("O", value="OTHER")
        .astype("string")
    )
    letter_grades = pd.cut(
        df[grade_num_col],
        # pandas' binning args here are bad if you want (standard!) left-inclusive bins
        # and *labels* for those bins; despite appearances, binning is like so:
        # [0, 0.7) => F, [0.7, 1.7) => D, [1.7, 2.7) => C, [2.7, 3.7) => B, [3.7, 4.0] => A
        bins=[0.0, 0.69, 1.69, 2.69, 3.69, 4.01],
        labels=["F", "D", "C", "B", "A"],
        right=True,
        include_lowest=True,
    ).astype("string")
    # NOTE: this assumes that "F" ("Fail") grades are equivalent to "F" letter grades
    return non_numeric_grades.combine_first(letter_grades)


def _grade_is_passing(grade: str, min_passing_grade: float) -> bool | None:
    if grade in NON_PASS_FAIL_GRADES:
        return None
    elif grade == "P":
        return True
    elif grade == "F":
        return False
    else:
        return float(grade) >= min_passing_grade
