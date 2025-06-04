import functools as ft
import typing as t

import numpy as np
import pandas as pd

from .... import utils
from ... import checkpoints
from . import shared


def compute_target(
    df: pd.DataFrame,
    *,
    intensity_time_limits: utils.types.IntensityTimeLimitsType,
    years_to_degree_col: str,
    num_terms_in_year: int = 4,
    max_term_rank: int | t.Literal["infer"] = "infer",
    student_id_cols: str | list[str] = "student_id",
    enrollment_intensity_col: str = "student_term_enrollment_intensity",
    enrollment_year_col: str = "year_of_enrollment_at_cohort_inst",
    term_is_pre_cohort_col: str = "term_is_pre_cohort",
    term_rank_col: str = "term_rank",
) -> pd.Series:
    """
    Compute *non* graduation target for each distinct student in ``df`` , for which
    intensity-specific time limits determine if graduation is "on-time"; null values
    are assigned when target can't be computed based on the time spanned in the dataset.

    Args:
        df: Student-term dataset.
        intensity_time_limits: Mapping of enrollment intensity value (e.g. "FULL-TIME")
            to the maximum number of years or terms considered to be an "on-time" graduation
            (e.g. [4.0, "year"], [12.0, "term"]), measured from the first term of enrollment
            within students' cohorts. To use the same time limit for all students,
            regardless of enrollment intensity, specify "*" as the only key.
        years_to_degree_col: Column whose values give the year of enrollment at cohort inst
            in which students *first* earned a particular degree; for example,
            "first_year_to_bachelors_at_cohort_inst" for Bachelors-seeking students.
            Note: If ``df`` has one row per student-term, it's assumed that years-to-degree
            values are the same across all rows, and we simply use the first.
        num_terms_in_year: Number of academic terms in one academic year,
            used to convert from term-based time limits to year-based time limits;
            default value assumes FALL, WINTER, SPRING, and SUMMER terms.
        max_term_rank: Maximum term rank value in the full dataset ``df`` , either inferred
            from ``df[term_rank_col]`` itself or as a manually specified value which
            may be different from the actual max value in ``df`` , depending on use case.
        student_id_cols: One or multiple columns uniquely identifying students.
        enrollment_intensity_col: Column whose values give students' "enrollment intensity"
            (usually either "FULL-TIME" or "PART-TIME"), for which the most common
            value per student is used when comparing against intensity-specific time limits.
        enrollment_year_col: Column whose values give students' "current" year of enrollment
            at the cohort inst as of the given row, used to filter rows to pre-graduation
            when determining students' most common enrollment intensity.
        term_is_pre_cohort_col
        term_rank_col: Column whose values give the absolute integer ranking of a given
            term within the full dataset ``df`` .

    See Also:
        - :func:`shared.get_students_with_max_target_term_in_dataset()`
        - :func:`checkpoints.pdp.first_student_terms_within_cohort()`
    """
    student_id_cols = utils.types.to_list(student_id_cols)
    # we want a target for every student in input df; this will ensure it
    df_distinct_students = df[student_id_cols].drop_duplicates(ignore_index=True)
    # get most common intensity value per student across all pre-graduation terms
    # and *first* years to degree value (we assume this is the same across all terms)
    # use this as reference data for computing target variable
    df_student_intensities = (
        df.loc[
            df[enrollment_year_col]
            .le(df[years_to_degree_col].astype("Int8"))
            .fillna(True),
            :,
        ]
        .groupby(by=student_id_cols, as_index=False)
        .agg(enrollment_intensity=(enrollment_intensity_col, _mode_aggfunc))
    )
    df_student_years_to_degree = df.groupby(by=student_id_cols, as_index=False).agg(
        years_to_degree=(years_to_degree_col, "first")
    )
    df_ref = pd.merge(
        df_student_intensities,
        df_student_years_to_degree,
        how="inner",
        on=student_id_cols,
    )
    # convert from term limits to year limits, as needed
    intensity_num_years = utils.misc.convert_intensity_time_limits(
        "year", intensity_time_limits, num_terms_in_year=num_terms_in_year
    )
    # compute all intensity/year boolean arrays separately
    # then combine with a logical OR
    targets = [
        (
            # enrollment intensity is equal to specified value or "*" given as intensity
            (df_ref["enrollment_intensity"].eq(intensity) | (intensity == "*"))
            & (
                # student graduated after max num years allowed
                (df_ref["years_to_degree"]).gt(num_years)
                # or never graduated at all
                | df_ref["years_to_degree"].isna()
            )
        )
        for intensity, num_years in intensity_num_years.items()
    ]
    target = np.logical_or.reduce(targets)
    # assign True to all students passing intensity/term condition(s) above
    df_target_true = (
        df_ref.loc[target, student_id_cols]
        .assign(target=True)
        .astype({"target": "boolean"})
    )
    # get all students for which a target label can accurately be computed
    # i.e. the data in df covers their last "on-time" graduation term
    df_labelable_students = shared.get_students_with_max_target_term_in_dataset(
        df,
        checkpoint=ft.partial(
            checkpoints.pdp.first_student_terms_within_cohort,
            term_is_pre_cohort_col=term_is_pre_cohort_col,
            student_id_cols=student_id_cols,
            sort_cols=term_rank_col,
            include_cols=(student_id_cols + [term_rank_col, enrollment_intensity_col]),
        ),
        intensity_time_limits=intensity_time_limits,
        max_term_rank=max_term_rank,
        num_terms_in_year=num_terms_in_year,
        student_id_cols=student_id_cols,
        enrollment_intensity_col=enrollment_intensity_col,
        term_rank_col=term_rank_col,
    )
    df_labeled = (
        # match positive labels to label-able students
        pd.merge(df_labelable_students, df_target_true, on=student_id_cols, how="left")
        # assign False to all label-able students not already assigned True
        .fillna({"target": False})
        # structure so student-ids as index, target as only column
        .set_index(student_id_cols)
    )
    df_all_student_targets = (
        # assign null target to all students
        df_distinct_students.assign(target=pd.Series(pd.NA, dtype="boolean"))
        # structure so student-ids as index, target as only column
        .set_index(student_id_cols)
    )
    # update null targets in-place with bool targets on matching student-id indexes
    df_all_student_targets.update(df_labeled)
    # #drop if target is uncalculable (null)
    df_all_student_targets["target"] = (
        df_all_student_targets["target"].astype("boolean").dropna()
    )
    # return as a series with target as values and student ids as index
    return df_all_student_targets.loc[:, "target"].dropna()


def _mode_aggfunc(ser: pd.Series, *, dropna: bool = True) -> object:
    mode = ser.mode(dropna=dropna)
    return mode.iat[0] if not mode.empty else pd.NA
