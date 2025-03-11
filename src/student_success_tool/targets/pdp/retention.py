import pandas as pd

from ... import utils


def compute_target(
    df: pd.DataFrame,
    *,
    student_id_cols: str | list[str] = "student_id",
    retention_col: str = "retention",
    na_value: bool = False,
) -> pd.Series:
    """
    Compute *non* retention target for each distinct student in ``df`` ,
    using the (logical converse) of PDP's definition for "retention".

    Args:
        df: Student-term dataset.
        student_id_cols: One or multiple columns uniquely identifying students.
        retention_col: Column whose values indicate whether or not a given student
            was retained, either as booleans or 0/1 integers.
            Note: If ``df`` has one row per student-term, it's assumed that retention
            values are the same across all rows, and we simply use the first.
        na_value: In case a student has a null retention value,
            this value should be assigned as their target value.

    References:
        - https://help.studentclearinghouse.org/pdp/knowledge-base/cohort-level-analysis-ready-file-data-dictionary/
    """
    student_id_cols = utils.types.to_list(student_id_cols)
    return (
        df[student_id_cols + [retention_col]]
        # we only want only one row per student
        # this assumes retention_col value is the same across rows for a given student
        .drop_duplicates(subset=student_id_cols, ignore_index=True)
        # we are predicting *non* retention
        .assign(target=lambda df: ~df[retention_col].astype("boolean"))
        # fill nulls and ensure we have "vanilla" bool values
        .fillna({"target": na_value})
        .astype({"target": "bool"})
        # return as a series with target as values and student ids as index
        .set_index(student_id_cols)
        .loc[:, "target"]
    )
