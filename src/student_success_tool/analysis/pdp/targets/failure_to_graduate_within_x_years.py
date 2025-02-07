import logging
import typing as t
from collections.abc import Collection
import pandas as pd
from student_success_tool.analysis.pdp.targets import shared
from student_success_tool.analysis.pdp import utils

LOGGER = logging.getLogger(__name__)


def make_labeled_dataset(
    df: pd.DataFrame,
    *,
    student_criteria: dict[str, object | Collection[object]],
    student_id_cols: str | list[str] = "student_guid",
    exclude_pre_cohort_terms: bool = True,
    term_is_pre_cohort_col: str = "term_is_pre_cohort",
    years_to_degree_col: str,
    intensity_time_lefts: list[tuple[str, float, t.Literal["year", "term"]]],
    max_term_rank: int,
    num_terms_in_year: int = 4,
    enrollment_intensity_col: str = "enrollment_intensity_first_term",
    term_rank_col: str = "term_rank",
    n: int, 
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
        intensity_time_lefts=intensity_time_lefts,
        max_term_rank=max_term_rank,
        num_terms_in_year=num_terms_in_year,
        enrollment_intensity_col=enrollment_intensity_col,
        term_rank_col=term_rank_col,
    )
    df_eligible_student_terms = pd.merge(
        df, df_eligible_students, on=student_id_cols, how="inner"
    )
    if exclude_pre_cohort_terms is True:
        df_features = shared.get_nth_student_terms_within_cohort(
            df_eligible_student_terms,
            student_id_cols=student_id_cols,
            term_is_pre_cohort_col=term_is_pre_cohort_col,
            sort_cols=term_rank_col,
            n=n,
            include_cols=None,
        )
    else:
        df_features = shared.get_nth_student_terms(
            df_eligible_student_terms,
            student_id_cols=student_id_cols,
            sort_cols=term_rank_col,
            n=n,
            include_cols=None,
        )
    df_targets = compute_target_variable(
        df_eligible_student_terms,
        student_id_cols=student_id_cols,
        years_to_degree_col=years_to_degree_col,
    )
    df_labeled = pd.merge(df_features, df_targets, on=student_id_cols, how="inner")
    return df_labeled


def compute_target_variable(  
    df: pd.DataFrame,  
    *,  
    student_id_cols: str | list[str] = "student_guid",  
    years_to_degree_col: str,  
    intensity_time_lefts: list[tuple[str, float, t.Literal["year", "term"]]], 
) -> pd.Series: 
    """
    Args:
        df
        student_id_cols
        years_to_degree_col
        intensity_time_lefts
    """ 
    student_id_cols = utils.to_list(student_id_cols) 
    intensity_num_terms = [(intensity, time if unit == "year" else time / num_terms_in_year)
                           for intensity, time, unit in intensity_time_lefts
                           ] 
    return (  
            df[student_id_cols + [years_to_degree_col]]  
            .drop_duplicates(subset=student_id_cols, ignore_index=True)  
            .assign(target=lambda df: (df[years_to_degree_col] > time) | (df[years_to_degree_col].isna()))  
            .fillna({"target": False})  
            .astype({"target": "bool"})  
            .set_index(student_id_cols)  
            .loc[:, "target"]  
)  

def select_eligible_students(  
    df: pd.DataFrame,
    *,
    student_criteria: dict[str, object | Collection[object]],
    student_id_cols: str | list[str] = "student_guid",
    intensity_time_lefts: list[tuple[str, float, t.Literal["year", "term"]]],
    max_term_rank: int,
    num_terms_in_year: int = 4,
    enrollment_intensity_col: str = "enrollment_intensity_first_term",
    term_rank_col: str = "term_rank",
) -> pd.DataFrame:
    """
    Args:
        df
        student_criteria
        student_id_cols
        intensity_time_lefts
        max_term_rank
        num_terms_in_year
        enrollment_intensity_col
        term_rank_col
    """
    nuq_students_in = df.groupby(by=student_id_cols, sort=False).ngroups
    df_students_by_criteria = shared.select_students_by_criteria(
        df, student_id_cols=student_id_cols, **student_criteria
    )
    df_students_by_time_left = (
        shared.select_students_by_time_left(
            df,
            student_id_cols=student_id_cols,
            intensity_time_lefts=intensity_time_lefts,
            max_term_rank=max_term_rank,
            num_terms_in_year=num_terms_in_year,
            enrollment_intensity_col=enrollment_intensity_col,
            term_rank_col=term_rank_col
        )
    )
    df_out = pd.merge(
        df_students_by_criteria,
        df_students_by_time_left,
        on=student_id_cols,
        how="inner",
    )
    nuq_students_out = df_out.groupby(by=student_id_cols, sort=False).ngroups
    shared._log_eligible_selection(nuq_students_in, nuq_students_out, "overall")
    return df_out