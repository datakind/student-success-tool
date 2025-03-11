import functools as ft
import re
import typing as t

import pandas as pd

from . import featurize, standardize


def preprocess_data(
    df_cohort: pd.DataFrame,
    df_course: pd.DataFrame,
    *,
    feature_params: dict,
    checkpoint_func: t.Callable[..., pd.DataFrame],
    checkpoint_params: t.Optional[dict] = None,
    selection_func: t.Callable[..., pd.Series],
    selection_params: t.Optional[dict] = None,
    target_func: t.Optional[t.Callable] = None,
    target_params: t.Optional[dict] = None,
    student_id_col: str = "student_id",
) -> pd.DataFrame:
    df_cohort = standardize.standardize_cohort_dataset(df_cohort)
    df_course = standardize.standardize_course_dataset(df_course)
    df_student_terms = featurize.featurize_student_terms(
        df_cohort, df_course, **feature_params
    )

    selected_students = selection_func(df_student_terms, **(selection_params or {}))
    df_ckpt = checkpoint_func(df_student_terms, **(checkpoint_params or {}))
    df_features = pd.merge(
        df_ckpt, pd.Series(selected_students), how="inner", on=student_id_col
    )

    if target_func is not None:
        targets = target_func(df_student_terms, **(target_params or {}))
        df_preproc = pd.merge(df_features, targets, how="inner", on=student_id_col)
    else:
        df_preproc = df_features

    df_preproc = _clean_up_preprocessed_data(df_preproc)

    return df_preproc


def _clean_up_preprocessed_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop a bunch of columns not needed or wanted for modeling, and set to null
    any values corresponding to time after a student's current year of enrollment.
    """
    return (
        # drop many columns that *should not* be included as features in a model
        df.pipe(
            standardize.drop_columns_safely,
            cols=[
                # metadata
                "institution_id",
                "cohort_id",
                "term_id",
                "academic_year",
                "cohort",
                # "academic_term",  # keeping this to see if useful
                # "cohort_term",  # keeping this to see if useful
                "term_rank",
                "term_rank_fall_spring",
                "term_is_fall_spring",
                # columns used to derive other features, but not features themselves
                # "grade",  # TODO: should this be course_grade?
                "course_ids",
                "course_subjects",
                "course_subject_areas",
                "min_student_term_rank",
                "min_student_term_rank_fall_spring",
                "sections_num_students_enrolled",
                "sections_num_students_passed",
                "sections_num_students_completed",
                "term_start_dt",
                "cohort_start_dt",
                # "outcome" variables / likely sources of data leakage
                "retention",
                "persistence",
                "years_to_bachelors_at_cohort_inst",
                "years_to_bachelor_at_other_inst",
                "years_to_associates_or_certificate_at_cohort_inst",
                "years_to_associates_or_certificate_at_other_inst",
                "years_of_last_enrollment_at_cohort_institution",
                "years_of_last_enrollment_at_other_institution",
            ],
        ).assign(
            # keep "first year to X credential at Y inst" values if they occurred
            # in any year prior to the current year of enrollment; otherwise, set to null
            # in this case, the values themselves represent years
            **{
                col: ft.partial(_mask_year_values_based_on_enrollment_year, col=col)
                for col in df.columns[
                    df.columns.str.contains(r"^first_year_to")
                ].tolist()
            }
            # keep values in "*_year_X" columns if they occurred in any year prior
            # to the current year of enrollment; otherwise, set to null
            # in this case, the columns themselves represent years
            | {
                col: ft.partial(_mask_year_column_based_on_enrollment_year, col=col)
                for col in df.columns[df.columns.str.contains(r"_year_\d$")].tolist()
            }
            # do the same thing, only by term for term columns: "*_term_X"
            | {
                col: ft.partial(_mask_term_column_based_on_enrollment_term, col=col)
                for col in df.columns[df.columns.str.contains(r"_term_\d$")].tolist()
            }
        )
    )


def _mask_year_values_based_on_enrollment_year(
    df: pd.DataFrame,
    *,
    col: str,
    enrollment_year_col: str = "year_of_enrollment_at_cohort_inst",
) -> pd.Series:
    return df[col].mask(df[col].ge(df[enrollment_year_col]), other=pd.NA)


def _mask_year_column_based_on_enrollment_year(
    df: pd.DataFrame,
    *,
    col: str,
    enrollment_year_col: str = "year_of_enrollment_at_cohort_inst",
) -> pd.Series:
    if match := re.search(r"_year_(?P<yr>\d)$", col):
        col_year = int(match.groupdict()["yr"])
    else:
        raise ValueError(f"column '{col}' does not end with '_year_NUM'")
    return df[col].mask(df[enrollment_year_col].le(col_year), other=pd.NA)


def _mask_term_column_based_on_enrollment_term(
    df: pd.DataFrame,
    *,
    col: str,
    enrollment_term_col: str = "cumnum_terms_enrolled",
) -> pd.Series:
    if match := re.search(r"_term_(?P<num>\d)$", col):
        col_term = int(match.groupdict()["num"])
    else:
        raise ValueError(f"column '{col}' does not end with '_term_NUM'")
    return df[col].mask(df[enrollment_term_col].lt(col_term), other=pd.NA)
