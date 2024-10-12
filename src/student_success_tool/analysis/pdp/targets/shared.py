import typing as t
from collections.abc import Collection

import numpy as np
import pandas as pd

from .. import utils


def select_students_by_criteria(
    df: pd.DataFrame,
    *,
    student_id_cols: str | list[str] = "student_guid",
    **criteria: object | Collection[object],
) -> pd.DataFrame:
    """
    Select distinct students in ``df`` that are eligible according to one or more criteria.

    Args:
        df: Student-term dataset. For example, as output by
            :func:`pdp.dataops.make_student_term_dataset()` .
        student_id_cols: Column(s) that uniquely identify students in ``df`` ,
            used to drop duplicates.
        **criteria: Column name in ``df`` mapped to one or multiple values
            that it must equal in order for the corresponding student to be considered
            "eligible". Multiple criteria are combined with a logical "AND". For example,
            ``enrollment_type="FIRST-TIME", enrollment_intensity_first_term=["FULL-TIME", "PART-TIME"]``
            selects first-time students who enroll at full- or part-time intensity.

    Warning:
        This assumes that students' eligibility criteria values are the same across all
        terms. If that assumption is violated, this code actually checks if the student
        satisfied all of the criteria during _any_ of their terms.
    """
    if not criteria:
        raise ValueError("one or more eligibility criteria must be specified")

    student_id_cols = utils.to_list(student_id_cols)
    is_eligibles = [
        df[key].isin(set(val))
        if utils.is_collection_but_not_string(val)
        else df[key].eq(val).fillna(value=False)  # type: ignore
        for key, val in criteria.items()
    ]
    is_eligible = np.logical_and.reduce(is_eligibles)
    return (
        df.loc[is_eligible, student_id_cols]
        # df is at student-term level; get student ids from first eligible terms only
        .drop_duplicates(ignore_index=True)
    )


def select_students_by_time_left(
    df: pd.DataFrame,
    *,
    intensity_time_lefts: list[tuple[str, float, t.Literal["year", "term"]]],
    max_term_rank: int,
    num_terms_in_year: int = 4,
    student_id_cols: str | list[str] = "student_guid",
    enrollment_intensity_col: str = "enrollment_intensity_first_term",
    term_rank_col: str = "term_rank",
) -> pd.DataFrame:
    """
    Select distinct students in ``df`` for which ``df`` includes enough time left
    (in terms or years) to cover the student's future "target" term,
    depending on the student's enrollment intensity.

    Args:
        df: Student-term dataset. See "Warning" below.
        intensity_time_lefts: One or more criteria that define the amount of time left
            for a given enrollment intensity for a student to be considered eligible,
            specified as a triple: (enrollment intensity, amount of time, time unit).
            For example:

                - ``[("FULL-TIME", 1.0, "year")]`` => for students enrolled at full-time
                  intensity, at least 1 year's worth of student-terms must be included
                  in the full dataset following the given student-term
                - ``[("FULL-TIME", 4.0, "term"), ("PART-TIME", 8.0, "term)]`` =>
                  for full-time students, at least 4 subsequent terms must be included
                  in the dataset; for part-time students, at least 8 subsequent terms

        max_term_rank: Maximum "term rank" value in full dataset. Not necessarily
            the maximum term rank included in ``df`` , depending on how it's filtered.
        num_terms_in_year: Number of (academic) terms in a(n academic) year.
            Used to convert times given in "year" units into times in "term" units.
        student_id_cols: Column(s) that uniquely identify students in ``df`` ,
            used to drop duplicates.
        enrollment_intensity_col: Column name in ``df`` with enrollment intensity values.
        term_rank_col: Column name in ``df`` with term rank values.

    Warning:
        This assumes that ``df`` only includes one student-term row per student; namely,
        the one from which time left should be measured. If that assumption is violated,
        this code actually checks if *any* of the student's terms occurred with
        enough time left for their particular enrollment intensity.
    """
    student_id_cols = utils.to_list(student_id_cols)
    intensity_num_terms = _compute_intensity_num_terms(
        intensity_time_lefts, num_terms_in_year
    )
    has_enough_terms_lefts = [
        (
            # enrollment intensity is equal to a specified value
            df[enrollment_intensity_col].eq(intensity)
            # and at least X terms left between reference term and max term in dataset
            & (max_term_rank - df[term_rank_col]).ge(num_terms)
        )
        for intensity, num_terms in intensity_num_terms
    ]
    has_enough_terms_left = np.logical_or.reduce(has_enough_terms_lefts)
    return (
        df.loc[has_enough_terms_left, student_id_cols]
        # df is at student-term level; get student ids from first eligible terms only
        .drop_duplicates(ignore_index=True)
    )


def get_first_student_terms(
    df: pd.DataFrame,
    *,
    student_id_cols: str | list[str] = "student_guid",
    sort_cols: str | list[str] = "term_rank",
    include_cols: t.Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    For each student, get the first row in ``df`` (in ascending order of ``sort_cols`` )
    and a configurable subset of columns.

    Args:
        df
        student_id_cols
        sort_cols
        include_cols
    """
    student_id_cols = utils.to_list(student_id_cols)
    sort_cols = utils.to_list(sort_cols)
    cols = (
        df.columns.tolist()
        if include_cols is None
        else list(
            utils.unique_elements_in_order(student_id_cols + sort_cols + include_cols)
        )
    )
    return (
        df.loc[:, cols]
        .sort_values(by=sort_cols, ascending=True)
        .groupby(by=student_id_cols, sort=False, as_index=False)
        .first()
    )


def get_first_student_terms_at_num_credits_earned(
    df: pd.DataFrame,
    *,
    min_num_credits: float,
    student_id_cols: str | list[str] = "student_guid",
    sort_cols: str | list[str] = "term_rank",
    num_credits_col: str = "num_credits_earned_cumsum",
    include_cols: t.Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    For each student, get the first row in ``df`` (in ascending order of ``sort_cols`` )
    for which their cumulative num credits earned is greater than or equal
    to the specified threshold value, and drop students that never earn enough.

    Args:
        df
        min_num_credits
        student_id_cols
        sort_cols
        num_credits_col
        include_cols
    """
    return get_first_student_terms(
        # exclude rows with insufficient num credits, so "first" meets our criteria here
        df.loc[df[num_credits_col].ge(min_num_credits), :],
        student_id_cols=student_id_cols,
        sort_cols=sort_cols,
        include_cols=include_cols,
    )


def _compute_intensity_num_terms(
    intensity_time_lefts: list[tuple[str, float, t.Literal["year", "term"]]],
    num_terms_in_year: int,
) -> list[tuple[str, float]]:
    """
    Compute the minimum number of terms required for a given enrollment intensity,
    converting values given in "years" into "terms" as needed.
    """
    return [
        (intensity, time if unit == "term" else time * num_terms_in_year)
        for intensity, time, unit in intensity_time_lefts
    ]