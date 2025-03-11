import typing as t

import numpy as np
import pandas as pd

from ... import utils


def compute_target(
    df: pd.DataFrame,
    *,
    intensity_time_limits: dict[str, tuple[float, t.Literal["year", "term"]]],
    num_terms_in_year: int = 4,
    student_id_cols: str | list[str] = "student_id",
    enrollment_intensity_col: str = "student_term_enrollment_intensity",
    years_to_degree_col: str = "first_year_to_bachelors_at_cohort_inst",
    enrollment_year_col: str = "year_of_enrollment_at_cohort_inst",
) -> pd.Series:
    """
    Compute *non* graduation target for each distinct student in ``df`` , for which
    intensity-specific time limits determine if graduation is "on-time".

    Args:
        df: Student-term dataset.
        intensity_time_limits: Mapping of enrollment intensity value (e.g. "FULL-TIME")
            to the maximum number of years or terms considered to be an "on-time" graduation
            (e.g. [4.0, "year"], [12.0, "term"]). Passing special "*" as the only key
            applies the corresponding time limits to all students, regardless of intensity.
        num_terms_in_year: Number of academic terms in one academic year,
            used to convert from term-based time limits to year-based time limits;
            default value assumes FALL, WINTER, SPRING, and SUMMER terms.
        student_id_cols: One or multiple columns uniquely identifying students.
        enrollment_intensity_col: Column whose values give students' "enrollment intensity"
            (usually either "FULL-TIME" or "PART-TIME"), for which the most common
            value per student is used when comparing against intensity-specific time limits.
        years_to_degree_col: Column whose values give the year of enrollment at cohort inst
            in which students *first* earned a particular degree; by default, we use
            PDP's standard "first years to a Bachelor's degree" column
            Note: If ``df`` has one row per student-term, it's assumed that years-to-degree
            values are the same across all rows, and we simply use the first.
        enrollment_year_col: Column whose values give students' "current" year of enrollment
            at the cohort inst as of the given row, used to filter rows to pre-graduation
            when determining students' most common enrollment intensity.
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
    return (
        # all students not assigned True, now assigned False
        pd.merge(df_distinct_students, df_target_true, on=student_id_cols, how="left")
        # fill nulls (i.e. non-True) with False and ensure we have "vanilla" bool values
        .fillna({"target": False})
        .astype({"target": "bool"})
        # return as a series with target as values and student ids as index
        .set_index(student_id_cols)
        .loc[:, "target"]
    )


def _mode_aggfunc(ser: pd.Series, *, dropna: bool = True) -> object:
    mode = ser.mode(dropna=dropna)
    return mode.iat[0] if not mode.empty else pd.NA
