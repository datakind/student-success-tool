import logging
import typing as t
from collections.abc import Collection

import numpy as np
import pandas as pd

from ... import utils

LOGGER = logging.getLogger(__name__)


def select_students_by_criteria(
    df: pd.DataFrame,
    *,
    student_id_cols: str | list[str] = "student_id",
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

    student_id_cols = utils.types.to_list(student_id_cols)
    nuq_students_in = df.groupby(by=student_id_cols, sort=False).ngroups
    is_eligibles = [
        df[key].isin(set(val))  # type: ignore
        if utils.types.is_collection_but_not_string(val)
        else df[key].eq(val).fillna(value=False)  # type: ignore
        for key, val in criteria.items()
    ]
    for (key, val), is_eligible_citerion in zip(criteria.items(), is_eligibles):
        nuq_students_criterion = (
            df.loc[is_eligible_citerion, student_id_cols]
            .groupby(by=student_id_cols, sort=False)
            .ngroups
        )
        _log_eligible_selection(
            nuq_students_in, nuq_students_criterion, f"{key}={val} criterion ..."
        )

    is_eligible = np.logical_and.reduce(is_eligibles)
    df_out = (
        df.loc[is_eligible, student_id_cols]
        # df is at student-term level; get student ids from first eligible terms only
        .drop_duplicates(ignore_index=True)
    )
    _log_eligible_selection(nuq_students_in, len(df_out), "all criteria")
    return df_out


def select_students_by_time_left(
    df: pd.DataFrame,
    *,
    intensity_time_lefts: list[tuple[str, float, t.Literal["year", "term"]]],
    max_term_rank: int,
    num_terms_in_year: int = 4,
    student_id_cols: str | list[str] = "student_id",
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
        1. This assumes that ``df`` only includes one student-term row per student; namely,
        the one from which time left should be measured. If that assumption is violated,
        this code actually checks if *any* of the student's terms occurred with
        enough time left for their particular enrollment intensity.
        2. Users should also always confirm with the school what to do / how to treat `NaN` intensities in the `enrollment_intensity_first_term` column, because the default in this pipeline drops them! Users should re-code these null values appropriately prior to running the pipeline if seeking to include them in the analysis/modeling dataset!**
    """
    student_id_cols = utils.types.to_list(student_id_cols)
    nuq_students_in = df.groupby(by=student_id_cols, sort=False).ngroups
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
    df_out = (
        df.loc[has_enough_terms_left, student_id_cols]
        # df is at student-term level; get student ids from first eligible terms only
        .drop_duplicates(ignore_index=True)
    )
    _log_eligible_selection(nuq_students_in, len(df_out), "time left")
    return df_out


def select_students_by_next_year_course_data(
    df: pd.DataFrame,
    *,
    student_id_cols: str | list[str] = "student_id",
    cohort_id_col: str = "cohort_id",
    term_id_col: str = "term_id",
) -> pd.DataFrame:
    """
    Select distinct students in ``df`` for which ``df`` includes any records
    for the *next* academic year after the students' cohort year; effectively,
    this drops all students in the cohort from the most recent course year.

    Args:
        df: Student-term dataset.
        student_id_cols
        cohort_id_col
        term_id_col
    """
    student_id_cols = utils.types.to_list(student_id_cols)
    nuq_students_in = df.groupby(by=student_id_cols, sort=False).ngroups
    max_term_year = (
        df[term_id_col].str.extract(r"^(\d{4})").astype("Int32").max().iat[0]
    )
    df_out = (
        df.groupby(by=student_id_cols, as_index=False)
        .agg(student_cohort_id=(cohort_id_col, "min"))
        .assign(
            student_cohort_year=lambda df: df["student_cohort_id"]
            .str.extract(r"^(\d{4})")
            .astype("Int32")
        )
        .loc[lambda df: df["student_cohort_year"] < max_term_year, student_id_cols]
    )
    _log_eligible_selection(nuq_students_in, len(df_out), "next year course data")
    return df_out


def get_first_student_terms(
    df: pd.DataFrame,
    *,
    student_id_cols: str | list[str] = "student_id",
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
    student_id_cols = utils.types.to_list(student_id_cols)
    sort_cols = utils.types.to_list(sort_cols)
    cols = (
        df.columns.tolist()
        if include_cols is None
        else list(
            utils.misc.unique_elements_in_order(
                student_id_cols + sort_cols + include_cols
            )
        )
    )
    return (
        df.loc[:, cols]
        .sort_values(by=sort_cols, ascending=True)
        # TODO: it seems like we should sort by student id in the returned data, no?
        # .sort_values(by=(student_id_cols + sort_cols), ascending=True)
        .groupby(by=student_id_cols, sort=False, as_index=False)
        .first()
    )


def get_first_student_terms_at_num_credits_earned(
    df: pd.DataFrame,
    *,
    min_num_credits: float,
    student_id_cols: str | list[str] = "student_id",
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


def get_first_student_terms_within_cohort(
    df: pd.DataFrame,
    *,
    student_id_cols: str | list[str] = "student_id",
    term_is_pre_cohort_col: str = "term_is_pre_cohort",
    sort_cols: str | list[str] = "term_rank",
    include_cols: t.Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    For each student, get the first row in ``df`` (in ascending order of ``sort_cols`` )
    for which the term occurred *within* the student's cohort, i.e. not prior to
    their official start of enrollment.

    Args:
        df
        student_id_cols
        cohort_id_col
        term_id_col
        term_rank_col
        sort_cols
        include_cols
    """
    return get_first_student_terms(
        # exclude rows that are "pre-cohort", so "first" meets our criteria here
        df.loc[df[term_is_pre_cohort_col].eq(False), :],
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


def get_nth_student_terms(
    df: pd.DataFrame,
    *,
    n: int,
    student_id_cols: str | list[str] = "student_id",
    sort_cols: str | list[str] = "term_rank",
    include_cols: t.Optional[list[str]] = None,
    term_is_pre_cohort_col: str = "term_is_pre_cohort",
    exclude_pre_cohort_terms: bool = True,
) -> pd.DataFrame:
    """
    For each student, get the nth row in ``df`` (in ascending order of ``sort_cols`` ). If `exclude_pre_cohort_col` is true, then for each student, we want to get the nth row in ``df`` (in ascending order of ``sort_cols`` ) for which the term occurred *within* the student's cohort, i.e. not prior to their official start of enrollment, and a configurable subset of columns.
    Ex. n = 0 gets the first term, and is equivalent to the functionality of get_first_student_terms(); n = 1 gets the second term, n = 2, gets the third term, so on and so forth.

    Args:
        df
        n
        student_id_cols
        sort_cols
        include_cols
        term_is_pre_cohort_col
        exclude_pre_cohort_terms
    """
    student_id_cols = utils.types.to_list(student_id_cols)
    sort_cols = utils.types.to_list(sort_cols)
    cols = (
        df.columns.tolist()
        if include_cols is None
        else list(
            utils.misc.unique_elements_in_order(
                student_id_cols + sort_cols + include_cols
            )
        )
    )
    # exclude rows that are "pre-cohort", so "nth" meets our criteria here
    df = (
        df.loc[df[term_is_pre_cohort_col].eq(False), :]
        if exclude_pre_cohort_terms is True
        else df.loc[:, cols]
    )
    return (
        df.sort_values(by=sort_cols, ascending=True)
        .groupby(by=student_id_cols, sort=False, as_index=False)
        .nth(n)
    )


def _log_eligible_selection(
    nunique_students_in: int, nunique_students_out: int, case: str
) -> None:
    LOGGER.info(
        "%s out of %s (%s%%) students selected as eligible by %s",
        nunique_students_out,
        nunique_students_in,
        round(100 * nunique_students_out / nunique_students_in, 1),
        case,
    )
