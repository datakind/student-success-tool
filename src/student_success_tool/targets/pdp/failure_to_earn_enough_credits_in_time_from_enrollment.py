import logging
import typing as t
from collections.abc import Collection

import numpy as np
import pandas as pd

from ... import features, utils
from ...preprocessing.pdp import dataops
from . import shared

LOGGER = logging.getLogger(__name__)


def make_labeled_dataset(
    df: pd.DataFrame,
    *,
    min_num_credits_checkin: float,
    min_num_credits_target: float,
    student_criteria: dict[str, object | Collection[object]],
    intensity_time_limits: list[tuple[str, float, t.Literal["year", "term"]]],
    student_id_cols: str | list[str] = "student_id",
    enrollment_intensity_col: str = "enrollment_intensity_first_term",
    num_credits_col: str = "num_credits_earned_cumsum",
    term_col: str = "academic_term",
    term_rank_col: str = "term_rank",
) -> pd.DataFrame:
    """
    Make a labeled dataset for modeling, where each row consists of features
    from eligible students' first qualifying term matched to computed target variables.

    Args:
        df
        min_num_credits_checkin
        min_num_credits_target
        student_criteria
        intensity_time_limits
        student_id_cols
        enrollment_intensity_col
        num_credits_col
        term_col
        term_rank_col

    See Also:
        - :func:`select_eligible_students()`
        - :func:`compute_target_variable()`
        - :func:`shared.get_first_student_terms_at_num_credits_earned()`
    """
    df_eligible_students = select_eligible_students(
        df,
        student_criteria=student_criteria,
        intensity_time_limits=intensity_time_limits,
        min_num_credits_checkin=min_num_credits_checkin,
        student_id_cols=student_id_cols,
        enrollment_intensity_col=enrollment_intensity_col,
        num_credits_col=num_credits_col,
        term_col=term_col,
        term_rank_col=term_rank_col,
    )
    df_eligible_student_terms = pd.merge(
        df, df_eligible_students, on=student_id_cols, how="inner"
    )
    df_features = shared.get_first_student_terms_at_num_credits_earned(
        df_eligible_student_terms,
        min_num_credits=min_num_credits_checkin,
        student_id_cols=student_id_cols,
        include_cols=None,
    )
    df_targets = compute_target_variable(
        df_eligible_student_terms,
        min_num_credits_target=min_num_credits_target,
        intensity_time_limits=intensity_time_limits,
        student_id_cols=student_id_cols,
    )
    df_labeled = pd.merge(df_features, df_targets, on=student_id_cols, how="inner")
    return df_labeled


def compute_target_variable(
    df: pd.DataFrame,
    *,
    min_num_credits_target: float,
    intensity_time_limits: list[tuple[str, float, t.Literal["year", "term"]]],
    student_id_cols: str | list[str] = "student_id",
    enrollment_intensity_col: str = "enrollment_intensity_first_term",
    num_credits_col: str = "num_credits_earned_cumsum",
    term_col: str = "academic_term",
    term_rank_col: str = "term_rank",
) -> pd.Series:
    """
    Args:
        df: Student-term dataset, with cumulative features added.
        min_num_credits_target: Minimum number of credits earned
            within specified time limits to be considered a *success* case.
        intensity_time_limits
        student_id_cols: Columns that uniquely identify students, used for grouping rows.
        enrollment_intensity_col
        num_credits_col
        term_col
        term_rank_col
    """
    student_id_cols = utils.types.to_list(student_id_cols)
    include_cols = [enrollment_intensity_col]
    # we want a target for every student in input df; this will ensure it
    df_distinct_students = df[student_id_cols].drop_duplicates(ignore_index=True)
    df_at_start = shared.get_first_student_terms(
        df,
        student_id_cols=student_id_cols,
        sort_cols=term_rank_col,
        include_cols=[enrollment_intensity_col],
    )
    df_at_target = shared.get_first_student_terms_at_num_credits_earned(
        df,
        min_num_credits=min_num_credits_target,
        student_id_cols=student_id_cols,
        include_cols=include_cols,
        sort_cols=term_rank_col,
        num_credits_col=num_credits_col,
    )
    df_at = pd.merge(
        df_at_start,
        df_at_target,
        on=student_id_cols,
        how="left",
        suffixes=("_start", "_target"),
    )
    num_terms_in_year = dataops.infer_num_terms_in_year(df[term_col])
    intensity_num_terms = shared._compute_intensity_num_terms(
        intensity_time_limits, num_terms_in_year
    )
    # compute all intensity/term boolean arrays separately
    # then combine with a logical OR
    trcol = term_rank_col  # hack, so logic below fits on lines
    targets = [
        (
            # enrollment intensity is equal to a specified value
            df_at[f"{enrollment_intensity_col}_start"].eq(intensity)
            & (
                (df_at[f"{trcol}_target"] - df_at[f"{trcol}_start"]).gt(num_terms)
                # or they *never* earned enough credits for target
                | df_at[f"{trcol}_target"].isna()
            )
        )
        for intensity, num_terms in intensity_num_terms
    ]
    target = np.logical_or.reduce(targets)
    # assign True to all students passing intensity/year condition(s) above
    df_target_true = (
        df_at.loc[target, student_id_cols]
        .assign(target=True)
        .astype({"target": "boolean"})
    )
    return (
        # all students not assigned True, now assigned False
        pd.merge(df_distinct_students, df_target_true, on=student_id_cols, how="left")
        .fillna(False)
        .astype({"target": "bool"})
        # TODO: do we want a series with student ids as index and target as values, or nah?
        .set_index(student_id_cols)
        .loc[:, "target"]
    )


def select_eligible_students(
    df: pd.DataFrame,
    *,
    student_criteria: dict[str, object | Collection[object]],
    intensity_time_limits: list[tuple[str, float, t.Literal["year", "term"]]],
    min_num_credits_checkin: float,
    student_id_cols: str | list[str] = "student_id",
    enrollment_intensity_col: str = "enrollment_intensity_first_term",
    num_credits_col: str = "num_credits_earned_cumsum",
    term_col: str = "academic_term",
    term_rank_col: str = "term_rank",
) -> pd.DataFrame:
    """
    Args:
        df
        student_criteria
        intensity_time_limits
        min_num_credits_checkin
        student_id_cols
        enrollment_intensity_col
        num_credits_col
        term_col
        term_rank_col
    """
    nuq_students_in = df.groupby(by=student_id_cols, sort=False).ngroups
    df_students_by_criteria = shared.select_students_by_criteria(
        df, student_id_cols=student_id_cols, **student_criteria
    )
    max_term_rank = df[term_rank_col].max()
    num_terms_in_year = dataops.infer_num_terms_in_year(df[term_col])
    df_students_by_num_creds = shared.get_first_student_terms_at_num_credits_earned(
        df,
        min_num_credits=min_num_credits_checkin,
        student_id_cols=student_id_cols,
        sort_cols=term_rank_col,
        num_credits_col=num_credits_col,
        include_cols=[],
    ).loc[:, utils.types.to_list(student_id_cols)]
    nuq_students_checkin = len(df_students_by_num_creds)
    shared._log_eligible_selection(
        nuq_students_in, nuq_students_checkin, "check-in credits earned"
    )
    df_ref = shared.get_first_student_terms(
        df,
        student_id_cols=student_id_cols,
        sort_cols=term_rank_col,
        include_cols=[enrollment_intensity_col],
    )
    df_students_by_time_left = shared.select_students_by_time_left(
        df_ref,
        intensity_time_lefts=intensity_time_limits,
        max_term_rank=max_term_rank,
        num_terms_in_year=num_terms_in_year,
        student_id_cols=student_id_cols,
        enrollment_intensity_col=enrollment_intensity_col,
        term_rank_col=term_rank_col,
    )
    df_out = features.pdp.shared.merge_many_dataframes(
        [df_students_by_criteria, df_students_by_num_creds, df_students_by_time_left],
        on=student_id_cols,
        how="inner",
    )
    nuq_students_out = df_out.groupby(by=student_id_cols, sort=False).ngroups
    shared._log_eligible_selection(nuq_students_in, nuq_students_out, "overall")
    return df_out
