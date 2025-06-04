import typing as t

import pandas as pd

from .... import utils
from . import shared


def compute_target(
    df: pd.DataFrame,
    *,
    max_academic_year: str | t.Literal["infer"] = "infer",
    student_id_cols: str | list[str] = "student_id",
    retention_col: str = "retention",
    cohort_id_col: str = "cohort_id",
    term_id_col: str = "term_id",
) -> pd.Series:
    """
    Compute *non* retention target for each distinct student in ``df`` ,
    using the (logical converse) of PDP's definition for "retention".

    Args:
        df: Student-term dataset.
        max_academic_year: Maximum academic year in the full dataset , either inferred
            from ``df[term_id_col]`` itself or as a manually specified value which
            may be different from the actual max value in ``df`` , depending on use case.
            Note: Value must be a string formatted as "YYYY[-YY]".
        student_id_cols: One or multiple columns uniquely identifying students.
        retention_col: Column whose values indicate whether or not a given student
            was retained, either as booleans or 0/1 integers.
            Note: If ``df`` has one row per student-term, it's assumed that retention
            values are the same across all rows, and we simply use the first.
        cohort_id_col: Column used to uniquely identify student cohorts.
        term_id_col: Colum used to uniquely identify academic terms.

    References:
        - https://help.studentclearinghouse.org/pdp/knowledge-base/cohort-level-analysis-ready-file-data-dictionary/
    """
    student_id_cols = utils.types.to_list(student_id_cols)
    # we need to compuate a target for every student in input df
    df_all_students = (
        df.loc[:, student_id_cols + [retention_col]]
        # df is student-term, so drop dupes to get one student per row
        .drop_duplicates(subset=student_id_cols, ignore_index=True)
    )
    df_all_student_targets = (
        # assign null target to all students; we'll fill in bool values later
        df_all_students.assign(target=pd.Series(pd.NA, dtype="boolean"))
        # structure so student-ids as index, target as only column
        .set_index(student_id_cols)
    )
    # get subset of students for which a target label can accurately be computed
    # i.e. the data in df covers their second year of enrollment
    df_labelable_students = shared.get_students_with_second_year_in_dataset(
        df,
        max_academic_year=max_academic_year,
        student_id_cols=student_id_cols,
        cohort_id_col=cohort_id_col,
        term_id_col=term_id_col,
    )
    df_labeled_students = (
        df_all_students.assign(target=lambda df: ~df[retention_col].astype("boolean"))
        .set_index(student_id_cols)
        .merge(df_labelable_students, how="inner", left_index=True, right_index=True)
    )
    # update null targets in-place with bool targets on matching student-id indexes
    df_all_student_targets.update(df_labeled_students)
    # drop if target is uncalculable (null)
    df_all_student_targets["target"] = (
        df_all_student_targets["target"].astype("boolean").dropna()
    )
    return df_all_student_targets.loc[:, "target"].dropna()

    # TODO: if we're confident that PDP computes null retention values to mean
    # second-year data not available in the data, then we can revert to this simpler form
    # return (
    #     df[student_id_cols + [retention_col]]
    #     # we only want only one row per student
    #     # this assumes retention_col value is the same across rows for a given student
    #     .drop_duplicates(subset=student_id_cols, ignore_index=True)
    #     # we are predicting *non* retention
    #     .assign(target=lambda df: ~df[retention_col].astype("boolean"))
    #     # return as a series with target as values and student ids as index
    #     .set_index(student_id_cols)
    #     .loc[:, "target"]
    # )
