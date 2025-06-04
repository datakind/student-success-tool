import logging
import typing as t

import pandas as pd

from ... import utils

LOGGER = logging.getLogger(__name__)


def nth_student_terms(
    df: pd.DataFrame,
    *,
    n: int = 0,
    type: t.Literal[
        "all", "num_credits_earned", "within_cohort", "enrollment_year"
    ] = "all",
    enrollment_year: t.Optional[int] = None,
    student_id_cols: str | list[str] = "student_id",
    sort_cols: str | list[str] = "term_rank",
    include_cols: t.Optional[list[str]] = None,
    term_is_pre_cohort_col: str = "term_is_pre_cohort",
    exclude_pre_cohort_terms: bool = True,
    term_is_core_col: str = "term_is_core",
    exclude_non_core_terms: bool = True,
    enrollment_year_col: t.Optional[str] = None,
    min_num_credits: t.Optional[float] = None,
    num_credits_col: t.Optional[str] = "num_credits_earned_cumsum",
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

    df_type = _get_type_df(
        df,
        type=type,
        enrollment_year_col=enrollment_year_col,
        enrollment_year=enrollment_year,
        min_num_credits=min_num_credits,
        term_is_pre_cohort_col=term_is_pre_cohort_col,
        num_credits_col=num_credits_col,
    )

    included_cols = _get_included_cols(
        df_type, student_id_cols, sort_cols, include_cols
    )

    if exclude_pre_cohort_terms:
        df_type = df_type[df_type[term_is_pre_cohort_col] == False]
    if exclude_non_core_terms:
        df_type = df_type[df_type[term_is_core_col] == True]

    df_nth = (
        df_type.loc[:, included_cols]
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


def _get_type_df(
    df: pd.DataFrame,
    type: t.Literal["all", "num_credits_earned", "within_cohort", "enrollment_year"],
    enrollment_year_col: t.Optional[str],
    enrollment_year: t.Optional[int],
    min_num_credits: t.Optional[float],
    term_is_pre_cohort_col: t.Optional[str] = "term_is_pre_cohort",
    num_credits_col: t.Optional[str] = "num_credits_earned_cumsum",
) -> pd.DataFrame:
    """
    Apply filtering on df based on type and flags.
    """
    df_type = df.copy()

    if type == "within_cohort":
        if term_is_pre_cohort_col not in df.columns:
            raise KeyError(f"'{term_is_pre_cohort_col}' not in DataFrame")
        df_type = df_type[~df_type[term_is_pre_cohort_col]]
    elif type == "enrollment_year":
        if enrollment_year_col not in df.columns:
            raise KeyError(f"'{enrollment_year_col}' not in DataFrame")
        df_type = df_type[df_type[enrollment_year_col] == enrollment_year]
    elif type == "num_credits_earned":
        if num_credits_col not in df.columns:
            raise KeyError("{num_credits_col} not in DataFrame")
        df_type = df_type.loc[df[num_credits_col].ge(min_num_credits), :]
    elif type == "all":
        pass
    else:
        raise ValueError(f"Invalid type: {type}")

    return df_type


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
