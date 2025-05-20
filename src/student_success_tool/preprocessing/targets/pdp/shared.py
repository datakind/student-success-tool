import logging
import typing as t

import numpy as np
import pandas as pd

from .... import utils

LOGGER = logging.getLogger(__name__)


def get_students_with_max_target_term_in_dataset(
    df: pd.DataFrame,
    *,
    checkpoint: pd.DataFrame | t.Callable[[pd.DataFrame], pd.DataFrame],
    intensity_time_limits: utils.types.IntensityTimeLimitsType,
    max_term_rank: int | t.Literal["infer"] = "infer",
    num_terms_in_year: int = 4,
    student_id_cols: str | list[str] = "student_id",
    enrollment_intensity_col: str = "student_term_enrollment_intensity",
    term_rank_col: str = "term_rank",
) -> pd.DataFrame:
    """
    Get set of students in ``checkpoint`` data for which ``df`` includes
    the last possible term within specified time limits, depending on the student's
    enrollment intensity in checkpoint term, such that a target could be known for sure.

    Args:
        df: Student-term dataset.
        checkpoint: "Checkpoint" from which time limits to target term are measured,
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
        # get student ids from first eligible terms only
        .drop_duplicates(subset=student_id_cols, ignore_index=True)
        .set_index(student_id_cols)
        # include relevant other columns, mostly just for debugging
        .assign(max_term_rank=max_term_rank)
        .astype("Int8")
    )
    _log_labelable_students(nuq_students_in, len(df_out))
    return df_out


def get_students_with_second_year_in_dataset(
    df: pd.DataFrame,
    *,
    max_academic_year: str | t.Literal["infer"] = "infer",
    student_id_cols: str | list[str] = "student_id",
    cohort_id_col: str = "cohort_id",
    term_id_col: str = "term_id",
) -> pd.DataFrame:
    """
    Get set of students in ``df`` for which ``df`` includes any records
    for the academic year after the students' cohort year (i.e. their second year);
    effectively, this excludes all students in the cohort from the most recent course year.

    Args:
        df: Student-term dataset.
        max_academic_year: Maximum academic year in the full dataset ``df`` , either inferred
            from ``df[term_id_col]`` itself or as a manually specified value which
            may be different from the actual max value in ``df`` , depending on use case.
            Note: Value must be a string formatted as "YYYY[-YY]".
        student_id_cols: One or multiple columns uniquely identifying students.
        cohort_id_col: Column used to uniquely identify student cohorts.
        term_id_col: Colum used to uniquely identify academic terms.
    """
    if max_academic_year == "infer":
        max_academic_year = _extract_year_from_id(df[term_id_col]).max()
    else:
        max_academic_year = _extract_year_from_id(pd.Series(max_academic_year)).iat[0]
    assert isinstance(max_academic_year, np.integer)  # type guard

    student_id_cols = utils.types.to_list(student_id_cols)
    nuq_students_in = df.groupby(by=student_id_cols, sort=False).ngroups
    df_out = (
        # distinct student ids set as index
        df.groupby(by=student_id_cols, as_index=True)
        # assume every value for a student's cohort id is the same, so "first" is fine
        .agg(student_cohort_id=(cohort_id_col, "first"))
        .assign(
            student_cohort_year=lambda df: _extract_year_from_id(
                df["student_cohort_id"]
            ),
            max_academic_year=max_academic_year,  # type: ignore
        )
        .loc[
            lambda df: df["student_cohort_year"].lt(max_academic_year),
            ["student_cohort_year", "max_academic_year"],
        ]
        .astype("Int32")
    )
    _log_labelable_students(nuq_students_in, len(df_out))
    return df_out


def _log_labelable_students(
    nunique_students_in: int, nunique_students_out: int
) -> None:
    LOGGER.info(
        "%s out of %s (%s%%) students in dataset are able to have targets computed",
        nunique_students_out,
        nunique_students_in,
        round(100 * nunique_students_out / nunique_students_in, 1),
    )


def _extract_year_from_id(ser: pd.Series) -> pd.Series:
    return ser.astype("string").str.extract(r"^(\d{4})", expand=False).astype("Int32")
