import functools as ft
import logging
import typing as t

import numpy as np
import pandas as pd

from . import constants, shared

LOGGER = logging.getLogger(__name__)


def aggregate_from_course_level_features(
    df: pd.DataFrame,
    *,
    student_term_id_cols: list[str],
    min_passing_grade: float = constants.DEFAULT_MIN_PASSING_GRADE,
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
        min_passing_grade: Minimum numeric grade considered by institution as "passing".
            Default value is 1.0, i.e. a "D" grade or better.
        key_course_subject_areas: List of course subject areas that are particularly
            relevant ("key") to the institution, such that features are computed to
            measure the number of courses falling within them per student-term.
        key_course_ids

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
        term_start_dt=("term_start_dt", "first"),
        term_rank=("term_rank", "first"),
        term_rank_core=("term_rank_core", "first"),
        term_rank_noncore=("term_rank_noncore", "first"),
        term_is_core=("term_is_core", "first"),
        term_is_noncore=("term_is_noncore", "first"),
        term_in_peak_covid=("term_in_peak_covid", "first"),
        term_program_of_study=("term_program_of_study", "first"),
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
            "course_grade",
        ],
    )

    agg_col_vals: list[tuple[str, t.Any | list[t.Any]]] = [
        ("core_course", "Y"),
        ("course_type", ["CC", "CD"]),
        ("course_level", [0, 1]),
        ("enrolled_at_other_institution_s", "Y"),
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
    df_dummy_equals = equal_cols_by_group(
        df=df_val_equals, grp_cols=student_term_id_cols
    )
    df_grade_aggs = multicol_grade_aggs_by_group(
        df, min_passing_grade=min_passing_grade, grp_cols=student_term_id_cols
    )
    return shared.merge_many_dataframes(
        [
            df_passthrough,
            df_aggs,
            df_val_equals,
            df_dummy_equals,
            df_dummies,
            df_grade_aggs,
        ],
        on=student_term_id_cols,
    )


def add_features(
    df: pd.DataFrame,
    *,
    min_num_credits_full_time: float = constants.DEFAULT_MIN_NUM_CREDITS_FULL_TIME,
) -> pd.DataFrame:
    """
    Compute various student-term-level features from aggregated course-level features
    joined to student-level features.

    Args:
        df
        min_num_credits_full_time: Minimum number of credits *attempted* per term
            for a student's enrollment intensity to be considered "full-time".
            Default value is 12.0.

    See Also:
        - :func:`aggregate_from_course_level_features()`
    """
    LOGGER.info("adding student-term features ...")
    nc_prefix = constants.NUM_COURSE_FEATURE_COL_PREFIX
    fc_prefix = constants.FRAC_COURSE_FEATURE_COL_PREFIX
    _num_course_cols = (
        [col for col in df.columns if col.startswith(f"{nc_prefix}_")]
        +
        # also include num-course cols to be added below
        [
            "num_courses_in_program_of_study_area_term_1",
            "num_courses_in_program_of_study_area_year_1",
            "num_courses_in_term_program_of_study_area",
        ]
    )
    num_frac_courses_cols = [
        (col, col.replace(f"{nc_prefix}_", f"{fc_prefix}_")) for col in _num_course_cols
    ]
    feature_name_funcs = (
        {
            "year_of_enrollment_at_cohort_inst": year_of_enrollment_at_cohort_inst,
            "student_has_earned_certificate_at_cohort_inst": ft.partial(
                student_earned_certificate, inst="cohort"
            ),
            "student_has_earned_certificate_at_other_inst": ft.partial(
                student_earned_certificate, inst="other"
            ),
            "term_is_pre_cohort": term_is_pre_cohort,
            "term_is_while_student_enrolled_at_other_inst": term_is_while_student_enrolled_at_other_inst,
            "term_program_of_study_area": term_program_of_study_area,
            "frac_credits_earned": shared.frac_credits_earned,
            "student_term_enrollment_intensity": ft.partial(
                student_term_enrollment_intensity,
                min_num_credits_full_time=min_num_credits_full_time,
            ),
            "num_courses_in_program_of_study_area_term_1": ft.partial(
                num_courses_in_study_area,
                study_area_col="student_program_of_study_area_term_1",
            ),
            "num_courses_in_program_of_study_area_year_1": ft.partial(
                num_courses_in_study_area,
                study_area_col="student_program_of_study_area_year_1",
            ),
            "num_courses_in_term_program_of_study_area": ft.partial(
                num_courses_in_study_area,
                study_area_col="term_program_of_study_area",
            ),
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
    df: pd.DataFrame,
    *,
    cohort_start_dt_col: str = "cohort_start_dt",
    term_start_dt_col: str = "term_start_dt",
) -> pd.Series:
    dts_diff = (df[term_start_dt_col].sub(df[cohort_start_dt_col])).dt.days
    return pd.Series(np.ceil((dts_diff + 1) / 365.25), dtype="Int8")


def student_earned_certificate(
    df: pd.DataFrame,
    *,
    inst: t.Literal["cohort", "other"],
    enrollment_year_col: str = "year_of_enrollment_at_cohort_inst",
) -> pd.Series:
    degree_year_cols = [
        f"first_year_to_certificate_at_{inst}_inst",
        f"years_to_latest_certificate_at_{inst}_inst",
    ]
    return df.loc[:, degree_year_cols].lt(df[enrollment_year_col], axis=0).any(axis=1)


def term_is_pre_cohort(
    df: pd.DataFrame,
    *,
    cohort_start_dt_col: str = "cohort_start_dt",
    term_start_dt_col: str = "term_start_dt",
) -> pd.Series:
    return df[term_start_dt_col].lt(df[cohort_start_dt_col]).astype("boolean")


# TODO: we could probably compute this directly, w/o an intermediate feature?
def term_is_while_student_enrolled_at_other_inst(
    df: pd.DataFrame, *, col: str = "num_courses_enrolled_at_other_institution_s_Y"
) -> pd.Series:
    return df[col].gt(0)


def term_program_of_study_area(
    df: pd.DataFrame, *, col: str = "term_program_of_study"
) -> pd.Series:
    return shared.extract_short_cip_code(df[col])


def num_courses_in_study_area(
    df: pd.DataFrame,
    *,
    study_area_col: str,
    course_subject_areas_col: str = "course_subject_areas",
    fill_value: str = "-1",
) -> pd.Series:
    return (
        pd.DataFrame(df[course_subject_areas_col].tolist(), dtype="string")
        .eq(df[study_area_col].fillna(fill_value), axis="index")
        .sum(axis="columns")
        .astype("Int8")
    )


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


def student_term_enrollment_intensity(
    df: pd.DataFrame,
    *,
    min_num_credits_full_time: float,
    num_credits_col: str = "num_credits_attempted",
) -> pd.Series:
    if df[num_credits_col].isna().any():
        LOGGER.warning(
            "%s null values found for '%s'; "
            "calculation of student_term_enrollment_intensity doesn't correctly handle nulls",
            df[num_credits_col].isna().sum(),
            num_credits_col,
        )
    return pd.Series(
        data=np.where(
            df[num_credits_col].ge(min_num_credits_full_time), "FULL-TIME", "PART-TIME"
        ),
        index=df.index,
        dtype="string",
    )


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


def equal_cols_by_group(
    df: pd.DataFrame,
    *,
    grp_cols: list[str],
) -> pd.DataFrame:
    """
    Compute dummy values for all of the num_course features

    Args:
        df
        grp_cols
    """
    num_prefix = constants.NUM_COURSE_FEATURE_COL_PREFIX
    dummy_prefix = constants.DUMMY_COURSE_FEATURE_COL_PREFIX

    course_subject_prefixes = [
        constants.NUM_COURSE_FEATURE_COL_PREFIX + "_course_id",
        constants.NUM_COURSE_FEATURE_COL_PREFIX + "_course_subject_area",
    ]

    dummy_cols = {
        col.replace(num_prefix, dummy_prefix, 1): df[col].ge(1)
        for col in df.columns
        if any(col.startswith(prefix) for prefix in course_subject_prefixes)
    }

    return df.assign(**dummy_cols).reindex(columns=grp_cols + list(dummy_cols.keys()))


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
        temp_col = (
            f"{col}_{'|'.join(str(item) for item in val)}"
            if isinstance(val, list)
            else f"{col}_{val}"
        )
        temp_col_series[temp_col] = shared.compute_values_equal(df[col], val)
    return (
        df.assign(**temp_col_series)
        .reindex(columns=grp_cols + list(temp_col_series.keys()))
        .groupby(by=grp_cols, observed=True, as_index=True)
        .agg("sum")
        .rename(columns=_rename_sum_by_group_col)
        .reset_index(drop=False)
    )


def _rename_sum_by_group_col(col: str) -> str:
    return f"{constants.NUM_COURSE_FEATURE_COL_PREFIX}_{col}"


def multicol_grade_aggs_by_group(
    df: pd.DataFrame,
    *,
    min_passing_grade: float,
    grp_cols: list[str],
    grade_col: str = "grade",
    grade_numeric_col: str = "course_grade_numeric",
    section_grade_numeric_col: str = "section_course_grade_numeric_mean",
) -> pd.DataFrame:
    return (
        df.loc[:, grp_cols + [grade_col, grade_numeric_col, section_grade_numeric_col]]
        # compute intermediate column values all at once, which is efficient
        .assign(
            course_grade_is_failing_or_withdrawal=ft.partial(
                _course_grade_is_failing_or_withdrawal,
                min_passing_grade=min_passing_grade,
                grade_col=grade_col,
                grade_numeric_col=grade_numeric_col,
            ),
            course_grade_above_section_avg=ft.partial(
                _course_grade_above_section_avg,
                grade_numeric_col=grade_numeric_col,
                section_grade_numeric_col=section_grade_numeric_col,
            ),
        )
        .groupby(by=grp_cols, observed=True, as_index=False)
        # so that we can efficiently aggregate those intermediate values per group
        .agg(
            num_courses_grade_is_failing_or_withdrawal=(
                "course_grade_is_failing_or_withdrawal",
                "sum",
            ),
            num_courses_grade_above_section_avg=(
                "course_grade_above_section_avg",
                "sum",
            ),
        )
    )


def _course_grade_is_failing_or_withdrawal(
    df: pd.DataFrame,
    min_passing_grade: float,
    grade_col: str = "grade",
    grade_numeric_col: str = "course_grade_numeric",
) -> pd.Series:
    return (
        df[grade_col].isin({"F", "W"})
        | df[grade_numeric_col].between(0.0, min_passing_grade, inclusive="left")
    )  # fmt: skip


def _course_grade_above_section_avg(
    df: pd.DataFrame,
    grade_numeric_col: str = "course_grade_numeric",
    section_grade_numeric_col: str = "section_course_grade_numeric_mean",
) -> pd.Series:
    return df[grade_numeric_col].gt(df[section_grade_numeric_col])
