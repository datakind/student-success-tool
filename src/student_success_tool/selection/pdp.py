import logging
import typing as t
from collections.abc import Collection

import numpy as np
import pandas as pd

from .. import utils

LOGGER = logging.getLogger(__name__)


def select_students_by_attributes(
    df: pd.DataFrame,
    *,
    student_id_cols: str | list[str] = "student_id",
    **criteria: object | Collection[object],
) -> pd.DataFrame:
    """
    Select distinct students in ``df`` with specified attributes,
    according to the logical "AND" of one or more selection criteria.

    Args:
        df: Student-term dataset.
        student_id_cols: Column(s) that uniquely identify students in ``df`` ,
            used to drop duplicates.
        **criteria: Column name in ``df`` mapped to one or multiple values
            that it must equal in order for the corresponding student to be selected.
            Multiple criteria are combined with a logical "AND". For example,
            ``enrollment_type="FIRST-TIME", enrollment_intensity_first_term=["FULL-TIME", "PART-TIME"]``
            selects first-time students who enroll at full- or part-time intensity.

    Warning:
        This assumes that students' attribute values are the same across all terms.
        If that assumption is violated, this code actually checks if the student
        satisfied all of the criteria during _any_ of their terms.
    """
    if not criteria:
        raise ValueError("one or more selection criteria must be specified")

    student_id_cols = utils.types.to_list(student_id_cols)
    nuq_students_in = df.groupby(by=student_id_cols, sort=False).ngroups
    is_selecteds = [
        df[key].isin(set(val))  # type: ignore
        if utils.types.is_collection_but_not_string(val)
        else df[key].eq(val).fillna(value=False)  # type: ignore
        for key, val in criteria.items()
    ]
    for (key, val), is_selected in zip(criteria.items(), is_selecteds):
        nuq_students_criterion = (
            df.loc[is_selected, student_id_cols]
            .groupby(by=student_id_cols, sort=False)
            .ngroups
        )
        _log_selection(
            nuq_students_in, nuq_students_criterion, f"{key}={val} criterion ..."
        )
    is_selected = np.logical_and.reduce(is_selecteds)
    df_selected = (
        df.loc[is_selected, student_id_cols + list(criteria.keys())]
        # df is at student-term level; get student ids from first selected terms only
        .drop_duplicates(subset=student_id_cols, ignore_index=True)
        .set_index(student_id_cols)
    )
    _log_selection(nuq_students_in, len(df_selected), "all criteria")
    return df_selected


def select_students_by_second_year_data(
    df: pd.DataFrame,
    *,
    student_id_cols: str | list[str] = "student_id",
    cohort_id_col: str = "cohort_id",
    term_id_col: str = "term_id",
) -> pd.DataFrame:
    """
    Select distinct students in ``df`` for which ``df`` includes any records
    for the academic year after the students' cohort year (i.e. their second year);
    effectively, this drops all students in the cohort from the most recent course year.

    Args:
        df: Student-term dataset.
        student_id_cols: Column(s) that uniquely identify students in ``df`` ,
            used to drop duplicates.
        cohort_id_col: Column used to uniquely identify student cohorts.
        term_id_col: Colum used to uniquely identify academic terms.

    TODO: maybe combine this with more general function below?
    """
    student_id_cols = utils.types.to_list(student_id_cols)
    nuq_students_in = df.groupby(by=student_id_cols, sort=False).ngroups
    max_academic_year = _extract_year_from_id(df[term_id_col]).max()
    df_selected = (
        df.groupby(by=student_id_cols, as_index=False)
        .agg(student_cohort_id=(cohort_id_col, "min"))
        .assign(
            student_cohort_year=lambda df: _extract_year_from_id(
                df["student_cohort_id"]
            ),
            max_academic_year=max_academic_year,
        )
        .loc[
            lambda df: df["student_cohort_year"].lt(max_academic_year),
            student_id_cols + ["student_cohort_year", "max_academic_year"],
        ]
        .set_index(student_id_cols)
        .astype("Int32")
    )
    _log_selection(nuq_students_in, len(df_selected), "second year data")
    return df_selected


def select_students_with_max_target_term_in_dataset(
    df: pd.DataFrame,
    *,
    checkpoint: pd.DataFrame | t.Callable[[pd.DataFrame], pd.DataFrame],
    intensity_time_limits: dict[str, tuple[float, t.Literal["year", "term"]]],
    max_term_rank: int | t.Literal["infer"] = "infer",
    num_terms_in_year: int = 4,
    student_id_cols: str | list[str] = "student_id",
    enrollment_intensity_col: str = "student_term_enrollment_intensity",
    term_rank_col: str = "term_rank",
) -> pd.DataFrame:
    """
    Select distinct students in ``checkpoint`` data for which ``df`` includes
    the last possible term within specified time limits, depending on the student's
    enrollment intensity in checkpoint term, such that a target could be known for sure.

    Args:
        df: Student-term dataset.
        checkpoint: "Checkpoint" from which time limits to target term are determined,
            typically either the first enrolled term or the first term above an intermediate
            number of credits earned; may be given as a data frame with one row per student,
            or as a callable that takes ``df`` as input and returns all checkpoint terms.
        intensity_time_limits: Mapping of enrollment intensity value (e.g. "FULL-TIME")
            to the maximum number of years or terms considered to be "on-time" for
            the target number of credits earned (e.g. [4.0, "year"], [12.0, "term"]),
            where the numeric values are for the time between "checkpoint" and "target"
            terms. Passing special "*" as the only key applies the corresponding time limits
            to all students, regardless of intensity.
        max_term_rank: Maximum term rank value in the full dataset ``df`` , either inferred
            from ``df[term_rank_col]`` itself or as a manually specified value which
            may be different from the actual max value in ``df`` , depending on use case.
        num_terms_in_year: Number of academic terms in one academic year,
            used to convert from year-based time limits to term-based time limits;
            default value assumes FALL, WINTER, SPRING, and SUMMER terms.
        student_id_cols: One or multiple columns uniquely identifying students.
        enrollment_intensity_col: Column whose values give students' "enrollment intensity"
            (usually either "FULL-TIME" or "PART-TIME"), for use in applying a time limit.
        term_rank_col: Column whose values give the absolute integer ranking of a given
            term within the full dataset ``df`` .
    """
    if max_term_rank == "infer":
        max_term_rank = int(df[term_rank_col].max())
    assert isinstance(max_term_rank, int)  # type guard

    student_id_cols = utils.types.to_list(student_id_cols)
    nuq_students_in = df.groupby(by=student_id_cols, sort=False).ngroups
    df_ckpt = (
        checkpoint.copy(deep=True)
        if isinstance(checkpoint, pd.DataFrame)
        else checkpoint(df)
    )
    if df_ckpt.groupby(by=student_id_cols).size().gt(1).any():
        raise ValueError("checkpoint df must include exactly 1 row per student")

    intensity_num_terms = utils.misc.convert_intensity_time_limits(
        "term", intensity_time_limits, num_terms_in_year=num_terms_in_year
    )
    if "*" in intensity_num_terms:
        df_ckpt = df_ckpt.assign(
            student_max_term_rank=lambda df: df[term_rank_col]
            + intensity_num_terms["*"]
        )
    else:
        df_ckpt = df_ckpt.assign(
            student_max_term_rank=lambda df: df[term_rank_col]
            + df[enrollment_intensity_col].map(intensity_num_terms)
        )
    df_out = (
        df_ckpt.loc[
            df_ckpt["student_max_term_rank"].le(max_term_rank),
            student_id_cols + ["student_max_term_rank"],
        ]
        # df is at student-term level; get student ids from first eligible terms only
        .drop_duplicates(ignore_index=True)
        .set_index(student_id_cols)
        .assign(max_term_rank=max_term_rank)
        .astype("Int8")
    )

    # target_term_availables = [
    #     (
    #         # enrollment intensity is equal to a specified value or "*" given as intensity
    #         (df_ckpt[enrollment_intensity_col].eq(intensity) | (intensity == "*"))
    #         # and at least X terms left between checkpoint term and max term in dataset
    #         & (max_term_rank - df_ckpt[term_rank_col]).ge(num_terms)
    #     )
    #     for intensity, num_terms in intensity_num_terms.items()
    # ]
    # target_term_available = np.logical_or.reduce(target_term_availables)
    # df_out = (
    #     df.loc[target_term_available, student_id_cols]
    #     # df is at student-term level; get student ids from first eligible terms only
    #     .drop_duplicates(ignore_index=True)
    # )
    _log_selection(nuq_students_in, len(df_out), "time left")
    return df_out


def _log_selection(
    nunique_students_in: int, nunique_students_out: int, case: str
) -> None:
    LOGGER.info(
        "%s out of %s (%s%%) students selected by %s",
        nunique_students_out,
        nunique_students_in,
        round(100 * nunique_students_out / nunique_students_in, 1),
        case,
    )


def _extract_year_from_id(ser: pd.Series) -> pd.Series:
    return ser.astype("string").str.extract(r"^(\d{4})", expand=False).astype("Int32")
