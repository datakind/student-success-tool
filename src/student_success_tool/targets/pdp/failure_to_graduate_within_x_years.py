import logging
import typing as t
from collections.abc import Collection

import numpy as np
import pandas as pd

from ... import features, utils
from . import shared

LOGGER = logging.getLogger(__name__)


def make_labeled_dataset(
    df: pd.DataFrame,
    *,
    student_criteria: dict[str, object | Collection[object]],
    n: int,
    student_id_cols: str | list[str] = "student_id",
    intensity_time_lefts: list[tuple[str, float, t.Literal["year", "term"]]],
    max_term_rank: int,
    num_terms_in_year: int = 2,
    exclude_pre_cohort_terms: bool = True,
    term_is_pre_cohort_col: str = "term_is_pre_cohort",
    years_to_degree_col: str = "first_year_to_bachelors_at_cohort_inst",
    enrollment_intensity_col: str = "enrollment_intensity_first_term",
    term_rank_col: str = "term_rank",
) -> pd.DataFrame:
    """
    Make a labeled dataset for modeling, where each row consists of features
    from eligible students' first term (within their cohort)
    matched to computed target variables.

    See Also:
        - :func:`select_eligible_students()`
        - :func:`compute_target_variable()`
        - :func:`shared.get_nth_student_terms()`
    """
    df_eligible_students = select_eligible_students(
        df,
        student_criteria=student_criteria,
        n=n,
        student_id_cols=student_id_cols,
        intensity_time_lefts=intensity_time_lefts,
        max_term_rank=max_term_rank,
        num_terms_in_year=num_terms_in_year,
        exclude_pre_cohort_terms=exclude_pre_cohort_terms,
        term_is_pre_cohort_col=term_is_pre_cohort_col,
        enrollment_intensity_col=enrollment_intensity_col,
        term_rank_col=term_rank_col,
    )
    df_eligible_student_terms = pd.merge(
        df, df_eligible_students, on=student_id_cols, how="inner"
    )
    df_features = shared.get_nth_student_terms(
        df_eligible_student_terms,
        n=n,
        student_id_cols=student_id_cols,
        sort_cols=term_rank_col,
        include_cols=None,
        term_is_pre_cohort_col=term_is_pre_cohort_col,
        exclude_pre_cohort_terms=exclude_pre_cohort_terms,
    )
    df_targets = compute_target_variable(
        df_eligible_student_terms,
        intensity_time_lefts=intensity_time_lefts,
        student_id_cols=student_id_cols,
        num_terms_in_year=num_terms_in_year,
        enrollment_intensity_col=enrollment_intensity_col,
        years_to_degree_col=years_to_degree_col,
    )
    df_labeled = pd.merge(df_features, df_targets, on=student_id_cols, how="inner")
    return df_labeled


def compute_target_variable(
    df: pd.DataFrame,
    *,
    intensity_time_lefts: list[tuple[str, float, t.Literal["year", "term"]]],
    student_id_cols: str | list[str] = "student_id",
    num_terms_in_year: int = 4,
    enrollment_intensity_col: str = "enrollment_intensity_first_term",
    years_to_degree_col: str = "first_year_to_bachelors_at_cohort_inst",
) -> pd.Series:
    """
    Args:
        df
        intensity_time_limits: Mapping of enrollment intensity value (e.g. "FULL-TIME")
            to the maximum number of years allowed to earn degree "in time" (e.g. 4).
        student_id_cols
        enrollment_intensity_col
        years_to_degree_col: Name of column giving the year in which students _first_
            earned a particular degree (by default, a Bachelor's degree).
        num_terms_in_year: Defined number of terms in one academic year.
    """
    student_id_cols = utils.types.to_list(student_id_cols)
    include_cols = student_id_cols + [enrollment_intensity_col, years_to_degree_col]
    # we want a target for every student in input df; this will ensure it
    df_distinct_students = df[student_id_cols].drop_duplicates(ignore_index=True)
    df_first_terms = shared.get_first_student_terms(
        df, student_id_cols=student_id_cols, include_cols=include_cols
    )
    intensity_time_limits = {
        intensity: time if unit == "year" else time / num_terms_in_year
        for intensity, time, unit in intensity_time_lefts
    }
    # compute all intensity/year boolean arrays separately
    # then combine with a logical OR
    targets = [
        (
            # enrollment intensity is equal to a specified value
            df_first_terms[enrollment_intensity_col].eq(intensity)
            & (
                # they graduated after max num years allowed
                (df_first_terms[years_to_degree_col]).gt(num_years)
                # or they *never* graduated
                | df_first_terms[years_to_degree_col].isna()
            )
        )
        for intensity, num_years in intensity_time_limits.items()
    ]
    target = np.logical_or.reduce(targets)
    df_target_true = (
        df_first_terms.loc[target, student_id_cols]
        .assign(target=True)
        .astype({"target": "boolean"})
    )
    return (
        # all students not assigned True, now assigned False
        pd.merge(df_distinct_students, df_target_true, on=student_id_cols, how="left")
        .fillna(False)
        .astype({"target": "bool"})
        .set_index(student_id_cols)
        .loc[:, "target"]
    )


def select_eligible_students(
    df: pd.DataFrame,
    *,
    student_criteria: dict[str, object | Collection[object]],
    n: int,
    student_id_cols: str | list[str] = "student_id",
    intensity_time_lefts: list[tuple[str, float, t.Literal["year", "term"]]],
    max_term_rank: int,
    num_terms_in_year: int = 4,
    exclude_pre_cohort_terms: bool = True,
    term_is_pre_cohort_col: str = "term_is_pre_cohort",
    enrollment_intensity_col: str = "enrollment_intensity_first_term",
    term_rank_col: str = "term_rank",
) -> pd.DataFrame:
    """
    Selecting eligible students by "criteria", of having atleast "time left" from their first cohort term, determined by the outcome variable, and by being enrolled in an "nth" term, determined by the prediction checkpoint.
    Ex. if the outcome variable is graduating witihin <=4 years, and the check-in is end of their first year (assuming 2 academic terms in a year), the student needs "time left" of atleast 4 years or more, meaning they have been enrolled for atleast 4 years following their cohort, and have completed at least 2 academic terms for the prediction checkpoint to be considered. We then predict on that second term data, representing the end of their first-year.
    Args:
        df
        student_criteria
        n
        student_id_cols
        intensity_time_lefts
        max_term_rank
        num_terms_in_year
        exclude_pre_cohort_terms
        term_is_pre_cohort_col
        enrollment_intensity_col
        term_rank_col
    """
    nuq_students_in = df.groupby(by=student_id_cols, sort=False).ngroups
    df_students_by_criteria = shared.select_students_by_criteria(
        df, student_id_cols=student_id_cols, **student_criteria
    )
    if exclude_pre_cohort_terms is True:
        df_ref = shared.get_first_student_terms_within_cohort(
            df,
            student_id_cols=student_id_cols,
            sort_cols=term_rank_col,
            include_cols=[enrollment_intensity_col],
        )
    else:
        df_ref = shared.get_first_student_terms(
            df,
            student_id_cols=student_id_cols,
            sort_cols=term_rank_col,
            include_cols=[enrollment_intensity_col],
        )
    df_students_by_checkin = shared.get_nth_student_terms(
        df,
        n=n,
        student_id_cols=student_id_cols,
        sort_cols=term_rank_col,
        include_cols=None,
        term_is_pre_cohort_col=term_is_pre_cohort_col,
        exclude_pre_cohort_terms=exclude_pre_cohort_terms,
    ).loc[:, utils.types.to_list(student_id_cols)]
    nuq_students_checkin = len(df_students_by_checkin)
    shared._log_eligible_selection(
        nuq_students_in, nuq_students_checkin, "check-in point"
    )
    df_students_by_time_left = shared.select_students_by_time_left(
        df_ref,
        student_id_cols=student_id_cols,
        intensity_time_lefts=intensity_time_lefts,
        max_term_rank=max_term_rank,
        num_terms_in_year=num_terms_in_year,
        enrollment_intensity_col=enrollment_intensity_col,
        term_rank_col=term_rank_col,
    )
    df_out = features.pdp.shared.merge_many_dataframes(
        [df_students_by_criteria, df_students_by_checkin, df_students_by_time_left],
        on=student_id_cols,
        how="inner",
    )
    nuq_students_out = df_out.groupby(by=student_id_cols, sort=False).ngroups
    shared._log_eligible_selection(nuq_students_in, nuq_students_out, "overall")
    return df_out
