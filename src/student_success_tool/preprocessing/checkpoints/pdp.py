import logging
import typing as t

import pandas as pd

from ... import utils

LOGGER = logging.getLogger(__name__)


def nth_student_terms(
    df: pd.DataFrame,
    *,
    n: int,
    student_id_cols: str | list[str] = "student_id",
    sort_cols: str | list[str] = "term_rank",
    include_cols: t.Optional[list[str]] = None,
    term_is_pre_cohort_col: str = "term_is_pre_cohort",
    exclude_pre_cohort_terms: bool = True,
    term_is_core_col: str = "term_is_core",
    exclude_non_core_terms: bool = True,
    enrollment_year_col: t.Optional[str] = None,
    valid_enrollment_year: t.Optional[int] = None,
) -> pd.DataFrame:
    """
    For each student, get the nth row in ``df`` (in ascending order of ``sort_cols`` ). If `exclude_pre_cohort_col` is true, then for each student, we want to get the nth row in ``df`` (in ascending order of ``sort_cols`` ) for which the term occurred *within* the student's cohort, i.e. not prior to their official start of enrollment, and a configurable subset of columns. This parameter can be set to False to ignore the student's cohort start date in choosing the `nth` term.
    Ex. n = 0 gets the first term, and is equivalent to the functionality of get_first_student_terms(); n = 1 gets the second term, n = 2, gets the third term, so on and so forth.
    The parameter "exclude_non_core_terms" ensures that we only count core terms in choosing thr `nth` core term. This parameter can be set to False to count all terms in choosing the `nth` term.
    Valid_enrollment_year is a parameter that if set, we drop nth term if it falls outside this enrollment year.

    Args:
        df
        n
        student_id_cols
        sort_cols
        include_cols
        term_is_pre_cohort_col
        exclude_pre_cohort_terms
        term_is_core_col
        exclude_non_core_terms
        enrollment_year_col
        valid_enrollment_year
    """
    student_id_cols = utils.types.to_list(student_id_cols)
    sort_cols = utils.types.to_list(sort_cols)
    included_cols = _get_included_cols(df, student_id_cols, sort_cols, include_cols)
    if exclude_pre_cohort_terms:
        df = df[df[term_is_pre_cohort_col] == False]
    if exclude_non_core_terms:
        df = df[df[term_is_core_col] == True]
    df_nth = (
        df.loc[:, included_cols]
        .sort_values(
            by=(student_id_cols + sort_cols), ascending=True, ignore_index=False
        )
        .groupby(by=student_id_cols)
        .nth(n)
    )
    if valid_enrollment_year is not None:
        if enrollment_year_col is None:
            raise ValueError(
                "Must specify 'enrollment_year_col' if 'valid_enrollment_year' is given."
            )
        if enrollment_year_col not in df.columns:
            raise KeyError(f"'{enrollment_year_col}' is not in the DataFrame.")
        df_nth = df_nth[df_nth[enrollment_year_col] == valid_enrollment_year]
    assert isinstance(df_nth, pd.DataFrame)
    return df_nth


def first_student_terms(
    df: pd.DataFrame,
    *,
    student_id_cols: str | list[str] = "student_id",
    sort_cols: str | list[str] = "term_rank",
    include_cols: t.Optional[list[str]] = None,
    term_is_pre_cohort_col: str = "term_is_pre_cohort",
    exclude_pre_cohort_terms: bool = False,
    term_is_core_col: str = "term_is_core",
    exclude_non_core_terms: bool = False,
) -> pd.DataFrame:
    """
    For each student, get the first (0th) row in ``df`` in ascending order of ``sort_cols`` ,
    and a configurable subset of columns.

    Args:
        df: Student-term dataset.
        student_id_cols: Column(s) that uniquely identify students in ``df`` .
        sort_cols: Column(s) used to sort students' terms, typically chronologically.
        include_cols
        term_is_pre_cohort_col: Column identifying if a term is pre-cohort
        exclude_pre_cohort_terms:
        term_is_core_col: Column identifying if a term is a core term, where core terms are by default FALL and SPRING
        exclude_non_core_terms:

    See Also:
        - :func:`nth_student_terms()`
    """
    return nth_student_terms(
        df,
        n=0,
        student_id_cols=student_id_cols,
        sort_cols=sort_cols,
        include_cols=include_cols,
        term_is_pre_cohort_col=term_is_pre_cohort_col,
        exclude_pre_cohort_terms=exclude_pre_cohort_terms,
        term_is_core_col=term_is_core_col,
        exclude_non_core_terms=exclude_non_core_terms,
    )


def last_student_terms(
    df: pd.DataFrame,
    *,
    student_id_cols: str | list[str] = "student_id",
    sort_cols: str | list[str] = "term_rank",
    include_cols: t.Optional[list[str]] = None,
    term_is_pre_cohort_col: str = "term_is_pre_cohort",
    exclude_pre_cohort_terms: bool = False,
    term_is_core_col: str = "term_is_core",
    exclude_non_core_terms: bool = False,
) -> pd.DataFrame:
    """
    For each student, get the last (-1th) row in ``df`` in ascending order of ``sort_cols`` ,
    and a configurable subset of columns.

    Args:
        df: Student-term dataset.
        student_id_cols: Column(s) that uniquely identify students in ``df`` .
        sort_cols: Column(s) used to sort students' terms, typically chronologically.
        include_cols
        term_is_pre_cohort_col: Column identifying if a term is pre-cohort
        exclude_pre_cohort_terms:
        term_is_core_col: Column identifying if a term is a core term, where core terms are by default FALL and SPRING
        exclude_non_core_terms:

    See Also:
        - :func:`nth_student_terms()`
    """
    return nth_student_terms(
        df,
        n=-1,
        student_id_cols=student_id_cols,
        sort_cols=sort_cols,
        include_cols=include_cols,
        term_is_pre_cohort_col=term_is_pre_cohort_col,
        exclude_pre_cohort_terms=exclude_pre_cohort_terms,
        term_is_core_col=term_is_core_col,
        exclude_non_core_terms=exclude_non_core_terms,
    )


def first_student_terms_at_num_credits_earned(
    df: pd.DataFrame,
    *,
    min_num_credits: float,
    num_credits_col: str = "cumsum_num_credits_earned",
    student_id_cols: str | list[str] = "student_id",
    sort_cols: str | list[str] = "term_rank",
    include_cols: t.Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    For each student, get the first row in ``df`` in ascending order of ``sort_cols``
    for which their cumulative num credits earned is greater than or equal
    to the specified threshold value.

    Args:
        df: Student-term dataset.
        min_num_credits
        num_credits_col
        student_id_cols
        sort_cols
        include_cols

    See Also:
        - :func:`first_student_terms()`

    Warning:
        Students that never earn at least ``min_num_credits`` are dropped.
    """
    return first_student_terms(
        # exclude rows with insufficient num credits, so "first" meets our criteria here
        df.loc[df[num_credits_col].ge(min_num_credits), :],
        student_id_cols=student_id_cols,
        sort_cols=sort_cols,
        include_cols=include_cols,
    )


def first_student_terms_within_cohort(
    df: pd.DataFrame,
    *,
    term_is_pre_cohort_col: str = "term_is_pre_cohort",
    student_id_cols: str | list[str] = "student_id",
    sort_cols: str | list[str] = "term_rank",
    include_cols: t.Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    For each student, get the first row in ``df`` in ascending order of ``sort_cols``
    for which the term occurred *within* the student's cohort, i.e. not prior to
    their official start of enrollment.

    Args:
        df: Student-term dataset.
        term_is_pre_cohort_col
        student_id_cols
        sort_cols
        include_cols

    See Also:
        - :func:`first_student_terms()`

    Warning:
        Students that only have pre-cohort enrollments are dropped,
        although such cases are very unlikely.
    """
    return first_student_terms(
        # exclude rows that are "pre-cohort"
        df.loc[df[term_is_pre_cohort_col].eq(False), :],
        student_id_cols=student_id_cols,
        sort_cols=sort_cols,
        include_cols=include_cols,
    )


def last_student_terms_in_enrollment_year(
    df: pd.DataFrame,
    *,
    enrollment_year: int,
    enrollment_year_col: str = "year_of_enrollment_at_cohort_inst",
    student_id_cols: str | list[str] = "student_id",
    sort_cols: str | list[str] = "term_rank",
    include_cols: t.Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    For each student, get the last row in ``df`` in ascending order of ``sort_cols``
    for which the term occurred during a particular year of students' enrollment;
    for example, ``enrollment_year=1`` => last terms in students' first year of enrollment.

    Args:
        df: Student-term dataset.
        enrollment_year
        enrollment_year_col
        student_id_cols
        sort_cols
        include_cols

    See Also:
        - :func:`last_student_terms()`

    Warning:
        Students that aren't enrolled for at least ``enrollment_year`` years are dropped.
    """
    return last_student_terms(
        # exclude rows that aren't in specified enrollment year
        df.loc[df[enrollment_year_col].eq(enrollment_year), :],
        student_id_cols=student_id_cols,
        sort_cols=sort_cols,
        include_cols=include_cols,
    )


def _get_included_cols(
    df: pd.DataFrame,
    student_id_cols: list[str],
    sort_cols: list[str],
    include_cols: t.Optional[list[str]] = None,
) -> list[str]:
    return (
        df.columns.tolist()
        if include_cols is None
        else list(
            utils.misc.unique_elements_in_order(
                student_id_cols + sort_cols + include_cols
            )
        )
    )
