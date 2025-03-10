import logging

import pandas as pd

LOGGER = logging.getLogger(__name__)


def add_features(df: pd.DataFrame, *, section_id_cols: list[str]) -> pd.DataFrame:
    """
    Compute section-level features from pdp course dataset w/ added course-level features,
    and add as columns to ``df`` .

    Args:
        df
        section_id_cols: Columns that uniquely identify sections, used to group course rows
            and merge section features back in.

    Note:
        Rows for which any value in ``section_id_cols`` is null won't have features
        computed. This is because such a group is "undefined" in some sense,
        so we can't know if the resulting features are accurate.

    See Also:
        - :func:`pdp.features.course.add_features()`
    """
    LOGGER.info("adding section features ...")
    df_section = (
        df.groupby(by=section_id_cols, as_index=False, observed=True, dropna=True)
        # generating named aggs via functions gives at least *some* testability
        .agg(
            section_num_students_enrolled=section_num_students_enrolled_col_agg(),
            section_num_students_passed=section_num_students_passed_col_agg(),
            section_num_students_completed=section_num_students_completed_col_agg(),
            section_course_grade_numeric_mean=section_course_grade_numeric_mean_col_agg(),
        )
        .astype({"section_course_grade_numeric_mean": "Float32"})
    )
    return pd.merge(df, df_section, on=section_id_cols, how="left")


def section_num_students_enrolled_col_agg(col: str = "student_id") -> pd.NamedAgg:
    return pd.NamedAgg(col, "count")


def section_num_students_passed_col_agg(col: str = "course_passed") -> pd.NamedAgg:
    return pd.NamedAgg(col, "sum")


def section_num_students_completed_col_agg(
    col: str = "course_completed",
) -> pd.NamedAgg:
    return pd.NamedAgg(col, "sum")


def section_course_grade_numeric_mean_col_agg(
    col: str = "course_grade_numeric",
) -> pd.NamedAgg:
    return pd.NamedAgg(col, "mean")
