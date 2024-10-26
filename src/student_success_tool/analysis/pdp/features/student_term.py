import functools as ft
import logging
import typing as t

import pandas as pd

from .. import constants
from . import shared

LOGGER = logging.getLogger(__name__)


def aggregate_from_course_level_features(
    df: pd.DataFrame,
    *,
    student_term_id_cols: list[str],
    key_course_subject_areas: t.Optional[list[str]] = None,
    key_course_ids: t.Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Aggregate course-level features up to student-term-level features
    by grouping on ``student_term_id_cols`` , then aggregating columns' values
    by specified functions or as dummy columns whose values are, in turn, summed.

    Args:
        df
        student_term_id_cols: Columns that uniquely identify student-terms,
            used to group rows in ``df`` and merge features back in.
        key_course_subject_areas: List of course subject areas that are particularly
            relevant ("key") to the institution, such that features are computed to
            measure the number of courses falling within them per student-term.

    See Also:
        - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html
        - https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#built-in-aggregation-methods

    Notes:
        Rows for which any value in ``student_term_id_cols`` is null are dropped
        and features aren't computed! This is because such a group is "undefined",
        so we can't know if the resulting features are correct.
    """
    LOGGER.info("aggregating course-level data to student-term-level features ...")
    df_grped = df.groupby(by=student_term_id_cols, observed=True, as_index=False)
    # pass through useful metadata and term features as-is
    # assumed to have the same values for every row per group
    df_passthrough = df_grped.agg(
        institution_id=("institution_id", "first"),
        academic_year=("academic_year", "first"),
        academic_term=("academic_term", "first"),
        term_rank=("term_rank", "first"),
        term_in_peak_covid=("term_in_peak_covid", "first"),
        term_rank_fall_spring=("term_rank_fall_spring", "first"),
        term_is_fall_spring=("term_is_fall_spring", "first"),
        term_course_begin_date_min=("term_course_begin_date_min", "first"),
        term_course_end_date_max=("term_course_end_date_max", "first"),
    )
    # various aggregations, with an eye toward cumulative features downstream
    df_aggs = df_grped.agg(
        num_courses=num_courses_col_agg(),
        num_courses_passed=num_courses_passed_col_agg(),
        num_courses_completed=num_courses_completed_col_agg(),
        num_credits_attempted=num_credits_attempted_col_agg(),
        num_credits_earned=num_credits_earned_col_agg(),
        course_ids=course_ids_col_agg(),
        course_subjects=course_subjects_col_agg(),
        course_subject_areas=course_subject_areas_col_agg(),
        course_id_nunique=course_id_nunique_col_agg(),
        course_subject_nunique=course_subject_nunique_col_agg(),
        course_subject_area_nunique=course_subject_area_nunique_col_agg(),
        course_level_mean=course_level_mean_col_agg(),
        course_level_std=course_level_std_col_agg(),
        course_grade_numeric_mean=course_grade_numeric_mean_col_agg(),
        course_grade_numeric_std=course_grade_numeric_std_col_agg(),
        section_num_students_enrolled_mean=section_num_students_enrolled_mean_col_agg(),
        section_num_students_enrolled_std=section_num_students_enrolled_std_col_agg(),
        sections_num_students_enrolled=sections_num_students_enrolled_col_agg(),
        sections_num_students_passed=sections_num_students_passed_col_agg(),
        sections_num_students_completed=sections_num_students_completed_col_agg(),
    )
    df_dummies = sum_dummy_cols_by_group(
        df,
        grp_cols=student_term_id_cols,
        agg_cols=[
            "course_type",
            "delivery_method",
            "math_or_english_gateway",
            "co_requisite_course",
            "course_instructor_employment_status",
            "course_instructor_rank",
            "course_level",
            "grade",  # TODO: only if this is actually categorical
        ],
    )

    agg_col_vals: list[tuple[str, t.Any | list[t.Any]]] = [
        ("core_course", "Y"),
        ("course_type", ["CC", "CD"]),
        ("enrolled_at_other_institution_s", "Y"),
        ("grade", ["0", "1", "F", "W"]),
    ]
    if key_course_subject_areas is not None:
        agg_col_vals.extend(
            ("course_subject_area", kcsa) for kcsa in key_course_subject_areas
        )
    if key_course_ids is not None:
        agg_col_vals.extend(("course_id", kc) for kc in key_course_ids)
    df_val_equals = sum_val_equal_cols_by_group(
        df, grp_cols=student_term_id_cols, agg_col_vals=agg_col_vals
    )
    df_applied = multicol_aggs_by_group(df, grp_cols=student_term_id_cols)
    return shared.merge_many_dataframes(
        [df_passthrough, df_aggs, df_val_equals, df_dummies, df_applied],
        on=student_term_id_cols,
    )


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute various student-term-level features from aggregated course-level features
    joined to student-level features.

    See Also:
        - :func:`aggregate_from_course_level_features()`
    """
    LOGGER.info("adding student-term features ...")
    nc_prefix = constants.NUM_COURSE_FEATURE_COL_PREFIX
    fc_prefix = constants.FRAC_COURSE_FEATURE_COL_PREFIX
    num_frac_courses_cols = [
        (col, col.replace(f"{nc_prefix}_", f"{fc_prefix}_"))
        for col in df.columns
        if col.startswith(f"{constants.NUM_COURSE_FEATURE_COL_PREFIX}_")
    ]
    feature_name_funcs = (
        {
            "year_of_enrollment_at_cohort_inst": year_of_enrollment_at_cohort_inst,
            "term_is_while_student_enrolled_at_other_inst": term_is_while_student_enrolled_at_other_inst,
            "frac_credits_earned": shared.frac_credits_earned,
        }
        | {
            fc_col: ft.partial(compute_frac_courses, numer_col=nc_col)
            for nc_col, fc_col in num_frac_courses_cols
        }
        | {
            "frac_sections_students_passed": ft.partial(
                compute_frac_sections_students,
                numer_col="sections_num_students_passed",
            ),
            "frac_sections_students_completed": ft.partial(
                compute_frac_sections_students,
                numer_col="sections_num_students_completed",
            ),
        }
        | {
            "student_pass_rate_above_sections_avg": ft.partial(
                student_rate_above_sections_avg,
                student_col="frac_courses_passed",
                sections_col="frac_sections_students_passed",
            ),
            "student_completion_rate_above_sections_avg": ft.partial(
                student_rate_above_sections_avg,
                student_col="frac_courses_completed",
                sections_col="frac_sections_students_completed",
            ),
        }
    )
    return df.assign(**feature_name_funcs)


def year_of_enrollment_at_cohort_inst(
    df: pd.DataFrame, *, cohort_col: str = "cohort", academic_col: str = "academic_year"
) -> pd.Series:
    return (
        df[academic_col]
        .str.extract(r"(?P<academic_yr>\d{4})-\d{2}")
        .astype("Int16")["academic_yr"]
        - df[cohort_col]
        .str.extract(r"(?P<cohort_yr>\d{4})-\d{2}")
        .astype("Int16")["cohort_yr"]
        + 1
    )


# TODO: we could probably compute this directly, w/o an intermediate feature?
def term_is_while_student_enrolled_at_other_inst(
    df: pd.DataFrame, *, col: str = "num_courses_enrolled_at_other_institution_s_Y"
) -> pd.Series:
    return df[col].gt(0)


def compute_frac_courses(
    df: pd.DataFrame, *, numer_col: str, denom_col: str = "num_courses"
) -> pd.Series:
    result = df[numer_col].div(df[denom_col])
    if not result.between(0.0, 1.0, inclusive="both").all():
        raise ValueError()
    return result


def compute_frac_sections_students(
    df: pd.DataFrame,
    *,
    numer_col: str,
    denom_col: str = "sections_num_students_enrolled",
) -> pd.Series:
    result = df[numer_col].div(df[denom_col])
    if not result.between(0.0, 1.0, inclusive="both").all():
        raise ValueError()
    return result


def student_rate_above_sections_avg(
    df: pd.DataFrame, *, student_col: str, sections_col: str
) -> pd.Series:
    return df[student_col].gt(df[sections_col])


def num_courses_grade_above_section_avg(
    df: pd.DataFrame,
    *,
    grade_col: str = "course_grade_numeric",
    section_grade_col: str = "section_course_grade_numeric_mean",
) -> object:
    # NOTE: pydata has gone off the rails....sum() is return np.int64 value, not an int
    # and mypy won't let me annotate with np.int64, so... "object" it is
    return df[grade_col].gt(df[section_grade_col]).sum()


def num_courses_col_agg(col: str = "course_id") -> pd.NamedAgg:
    return pd.NamedAgg(col, "count")


def num_courses_passed_col_agg(col: str = "course_passed") -> pd.NamedAgg:
    return pd.NamedAgg(col, "sum")


def num_courses_completed_col_agg(col: str = "course_completed") -> pd.NamedAgg:
    return pd.NamedAgg(col, "sum")


def num_credits_attempted_col_agg(
    col: str = "number_of_credits_attempted",
) -> pd.NamedAgg:
    return pd.NamedAgg(col, "sum")


def num_credits_earned_col_agg(col: str = "number_of_credits_earned") -> pd.NamedAgg:
    return pd.NamedAgg(col, "sum")


def course_ids_col_agg(col: str = "course_id") -> pd.NamedAgg:
    return pd.NamedAgg(col, _agg_values_in_list)


def course_subjects_col_agg(col: str = "course_cip") -> pd.NamedAgg:
    return pd.NamedAgg(col, _agg_values_in_list)


def course_subject_areas_col_agg(col: str = "course_subject_area") -> pd.NamedAgg:
    return pd.NamedAgg(col, _agg_values_in_list)


def _agg_values_in_list(ser: pd.Series) -> list:
    result = ser.tolist()
    assert isinstance(result, list)  # type guard
    return result


def course_id_nunique_col_agg(col: str = "course_id") -> pd.NamedAgg:
    return pd.NamedAgg(col, "nunique")


def course_subject_nunique_col_agg(col: str = "course_cip") -> pd.NamedAgg:
    return pd.NamedAgg(col, "nunique")


def course_subject_area_nunique_col_agg(
    col: str = "course_subject_area",
) -> pd.NamedAgg:
    return pd.NamedAgg(col, "nunique")


def course_level_mean_col_agg(col: str = "course_level") -> pd.NamedAgg:
    return pd.NamedAgg(col, "mean")


def course_level_std_col_agg(col: str = "course_level") -> pd.NamedAgg:
    return pd.NamedAgg(col, "std")


def course_grade_numeric_mean_col_agg(col: str = "course_grade_numeric") -> pd.NamedAgg:
    return pd.NamedAgg(col, "mean")


def course_grade_numeric_std_col_agg(col: str = "course_grade_numeric") -> pd.NamedAgg:
    return pd.NamedAgg(col, "std")


def section_num_students_enrolled_mean_col_agg(
    col: str = "section_num_students_enrolled",
) -> pd.NamedAgg:
    return pd.NamedAgg(col, "mean")


def section_num_students_enrolled_std_col_agg(
    col: str = "section_num_students_enrolled",
) -> pd.NamedAgg:
    return pd.NamedAgg(col, "std")


def sections_num_students_enrolled_col_agg(
    col: str = "section_num_students_enrolled",
) -> pd.NamedAgg:
    return pd.NamedAgg(col, "sum")


def sections_num_students_passed_col_agg(
    col: str = "section_num_students_passed",
) -> pd.NamedAgg:
    return pd.NamedAgg(col, "sum")


def sections_num_students_completed_col_agg(
    col: str = "section_num_students_completed",
) -> pd.NamedAgg:
    return pd.NamedAgg(col, "sum")


def sum_dummy_cols_by_group(
    df: pd.DataFrame, *, grp_cols: list[str], agg_cols: list[str]
) -> pd.DataFrame:
    """
    Compute dummy values for all ``agg_cols`` in ``df`` , then group by ``grp_cols``
    and aggregate by "sum" to get the number of values for each dummy value.

    Args:
        df
        grp_cols
        agg_cols
    """
    return (
        pd.get_dummies(
            df[grp_cols + agg_cols],
            columns=agg_cols,
            sparse=False,
            dummy_na=False,
            drop_first=False,
        )
        .groupby(by=grp_cols, observed=True, as_index=True)
        .agg("sum")
        .rename(columns=_rename_sum_by_group_col)
        .reset_index(drop=False)
    )


def sum_val_equal_cols_by_group(
    df: pd.DataFrame,
    *,
    grp_cols: list[str],
    agg_col_vals: list[tuple[str, t.Any]],
) -> pd.DataFrame:
    """
    Compute equal to specified values for all ``agg_col_vals`` in ``df`` ,
    then group by ``grp_cols`` and aggregate with a "sum".

    Args:
        df
        grp_cols
        agg_col_vals
    """
    temp_col_series = {}
    for col, val in agg_col_vals:
        # make multi-value col names nicer to read
        temp_col = f"{col}_{'|'.join(val)}" if isinstance(val, list) else f"{col}_{val}"
        temp_col_series[temp_col] = shared.compute_values_equal(df[col], val)
    return (
        df.assign(**temp_col_series)
        .reindex(columns=grp_cols + list(temp_col_series.keys()))
        .groupby(by=grp_cols, observed=True, as_index=True)
        .agg("sum")
        .rename(columns=_rename_sum_by_group_col)
        .reset_index(drop=False)
    )


def multicol_aggs_by_group(
    df: pd.DataFrame,
    *,
    grp_cols: list[str],
    grade_col: str = "course_grade_numeric",
    section_grade_col: str = "section_course_grade_numeric_mean",
) -> pd.DataFrame:
    df_grped = df.groupby(by=grp_cols, observed=True, as_index=False)
    df_applied = df_grped.apply(
        num_courses_grade_above_section_avg,
        grade_col=grade_col,
        section_grade_col=section_grade_col,
        include_groups=False,
    )
    assert isinstance(df_applied, pd.DataFrame)  # type guard
    # pandas does not give a shit about type guards, ignoring the complaint below
    return df_applied.rename(columns={None: "num_courses_grade_above_section_avg"})  # type: ignore


def _rename_sum_by_group_col(col: str) -> str:
    return f"{constants.NUM_COURSE_FEATURE_COL_PREFIX}_{col}"
