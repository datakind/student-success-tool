import functools as ft
import logging
import typing as t

import pandas as pd

from . import shared

LOGGER = logging.getLogger(__name__)

# TODO: rename feature?
# student_program_of_study_changed_first_year => student_program_of_study_changed_year_1


def add_features(df: pd.DataFrame, *, institution_state: str) -> pd.DataFrame:
    """
    Compute student-level features from a standardized cohort dataset,
    and add as columns to ``df`` .

    Args:
        df
        institution_state: Standard, 2-letter abbreviation for the state in which
            the cohort institution is located.
    """
    LOGGER.info("adding student features ...")
    feature_name_funcs: dict[str, t.Callable[[pd.DataFrame], pd.Series]] = {
        # NOTE: it's likely that this the "most recent" enrollments info
        # comes from "the future" wrt predictions, so we can't include as model features
        # holding space here in case PDP tells us otherwise!
        # "student_has_prior_enrollment_at_other_inst": ft.partial(
        #     student_has_prior_enrollment_at_other_inst
        # ),
        # "student_prior_enrollment_at_other_inst_was_in_state": ft.partial(
        #     student_prior_enrollment_at_other_inst_was_in_state,
        #     institution_state=institution_state,
        # ),
        "student_program_of_study_area_term_1": ft.partial(
            student_program_of_study_area, col="program_of_study_term_1"
        ),
    }
    # in case of num_terms_checkin == 1, these features can't be computed
    # bc a necessary raw column gets dropped in the standardize function
    if "program_of_study_year_1" in df.columns:
        feature_name_funcs |= {
            "student_program_of_study_area_year_1": ft.partial(
                student_program_of_study_area, col="program_of_study_year_1"
            ),
            "student_program_of_study_changed_first_year": ft.partial(
                student_program_of_study_changed_first_year,
                term_col="program_of_study_term_1",
                year_col="program_of_study_year_1",
            ),
            "student_program_of_study_area_changed_first_year": ft.partial(
                student_program_of_study_changed_first_year,
                term_col="student_program_of_study_area_term_1",
                year_col="student_program_of_study_area_year_1",
            ),
            "diff_gpa_year_1_to_term_1": ft.partial(
                diff_gpa_year_1_to_term_1,
                term_col="gpa_group_term_1",
                year_col="gpa_group_year_1",
            ),
        }
    return df.assign(**feature_name_funcs)


def student_has_prior_enrollment_at_other_inst(
    df: pd.DataFrame,
    *,
    col: str = "most_recent_last_enrollment_at_other_institution_state",
) -> pd.Series:
    return df[col].astype("string").notna()


def student_prior_enrollment_at_other_inst_was_in_state(
    df: pd.DataFrame,
    *,
    col: str = "most_recent_last_enrollment_at_other_institution_state",
    institution_state: str,
) -> pd.Series:
    return df[col].astype("string").eq(institution_state).astype("boolean")


def student_program_of_study_area(df: pd.DataFrame, *, col: str) -> pd.Series:
    return shared.extract_short_cip_code(df[col])


def student_program_of_study_changed_first_year(
    df: pd.DataFrame, *, term_col: str, year_col: str
) -> pd.Series:
    return df[term_col].ne(df[year_col]).astype("boolean")


def diff_gpa_year_1_to_term_1(
    df: pd.DataFrame, *, term_col: str = "gpa_group_term_1", year_col: str = "gpa_group_year_1"
) -> pd.Series:
    return df[year_col] - df[term_col]
