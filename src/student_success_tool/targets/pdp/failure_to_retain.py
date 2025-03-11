import logging
from collections.abc import Collection

import pandas as pd

from ... import utils
from . import shared

LOGGER = logging.getLogger(__name__)


def make_labeled_dataset(
    df: pd.DataFrame,
    *,
    student_criteria: dict[str, object | Collection[object]],
    exclude_pre_cohort_terms: bool = True,
    student_id_cols: str | list[str] = "student_id",
    cohort_id_col: str = "cohort_id",
    term_id_col: str = "term_id",
    term_is_pre_cohort_col: str = "term_is_pre_cohort",
    term_rank_col: str = "term_rank",
    retention_col: str = "retention",
) -> pd.DataFrame:
    """
    Make a labeled dataset for modeling, where each row consists of features
    from eligible students' first term (within their cohort)
    matched to computed target variables.

    See Also:
        - :func:`select_eligible_students()`
        - :func:`compute_target_variable()`
        - :func:`shared.get_first_student_terms_within_cohort()`
    """
    df_eligible_students = select_eligible_students(
        df,
        student_criteria=student_criteria,
        student_id_cols=student_id_cols,
        cohort_id_col=cohort_id_col,
        term_id_col=term_id_col,
    )
    df_eligible_student_terms = pd.merge(
        df, df_eligible_students, on=student_id_cols, how="inner"
    )
    if exclude_pre_cohort_terms is True:
        df_features = shared.get_first_student_terms_within_cohort(
            df_eligible_student_terms,
            student_id_cols=student_id_cols,
            term_is_pre_cohort_col=term_is_pre_cohort_col,
            sort_cols=term_rank_col,
            include_cols=None,
        )
    else:
        df_features = shared.get_first_student_terms(
            df_eligible_student_terms,
            student_id_cols=student_id_cols,
            sort_cols=term_rank_col,
            include_cols=None,
        )
    df_targets = compute_target_variable(
        df_eligible_student_terms,
        student_id_cols=student_id_cols,
        retention_col=retention_col,
    )
    df_labeled = pd.merge(df_features, df_targets, on=student_id_cols, how="inner")
    return df_labeled


def compute_target_variable(
    df: pd.DataFrame,
    *,
    student_id_cols: str | list[str] = "student_id",
    retention_col: str = "retention",
) -> pd.Series:
    """
    Args:
        df
        student_id_cols
        retention_col
    """
    student_id_cols = utils.types.to_list(student_id_cols)
    return (
        df[student_id_cols + [retention_col]]
        .drop_duplicates(subset=student_id_cols, ignore_index=True)
        # recall that we are prediction *non* retention
        .assign(target=lambda df: ~df[retention_col])
        .drop(columns=retention_col)
        .fillna({"target": False})
        .astype({"target": "bool"})
        # TODO: do we want a series with student ids as index and target as values, or nah?
        .set_index(student_id_cols)
        .loc[:, "target"]
    )


def select_eligible_students(
    df: pd.DataFrame,
    *,
    student_criteria: dict[str, object | Collection[object]],
    student_id_cols: str | list[str] = "student_id",
    cohort_id_col: str = "cohort_id",
    term_id_col: str = "term_id",
) -> pd.DataFrame:
    """
    Args:
        df
        student_criteria
        student_id_cols
        cohort_id_col
        term_id_col
    """
    nuq_students_in = df.groupby(by=student_id_cols, sort=False).ngroups
    df_students_by_criteria = shared.select_students_by_criteria(
        df, student_id_cols=student_id_cols, **student_criteria
    )
    df_students_by_next_year_course_data = (
        shared.select_students_by_next_year_course_data(
            df,
            student_id_cols=student_id_cols,
            cohort_id_col=cohort_id_col,
            term_id_col=term_id_col,
        )
    )
    df_out = pd.merge(
        df_students_by_criteria,
        df_students_by_next_year_course_data,
        on=student_id_cols,
        how="inner",
    )
    nuq_students_out = df_out.groupby(by=student_id_cols, sort=False).ngroups
    shared._log_eligible_selection(nuq_students_in, nuq_students_out, "overall")
    return df_out
