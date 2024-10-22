import functools as ft
import logging

import pandas as pd

from . import shared

LOGGER = logging.getLogger(__name__)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute student-level features from a standardized cohort dataset,
    and add as columns to ``df`` .

    Args:
        df
    """
    LOGGER.info("adding student features ...")
    return df.assign(
        student_program_of_study_area_term_1=ft.partial(
            student_program_of_study_area, col="program_of_study_term_1"
        ),
        student_program_of_study_area_year_1=ft.partial(
            student_program_of_study_area, col="program_of_study_year_1"
        ),
        student_program_of_study_changed_term_1_to_year_1=ft.partial(
            student_program_of_study_changed_term_1_to_year_1,
            term_col="program_of_study_term_1",
            year_col="program_of_study_year_1",
        ),
        student_program_of_study_area_changed_term_1_to_year_1=ft.partial(
            student_program_of_study_changed_term_1_to_year_1,
            term_col="student_program_of_study_area_term_1",
            year_col="student_program_of_study_area_year_1",
        ),
        diff_gpa_term_1_to_year_1=ft.partial(diff_gpa_term_1_to_year_1),
    )


def student_program_of_study_area(df: pd.DataFrame, *, col: str) -> pd.Series:
    return shared.extract_short_cip_code(df[col])


def student_program_of_study_changed_term_1_to_year_1(
    df: pd.DataFrame, *, term_col: str, year_col: str
) -> pd.Series:
    return df[term_col].ne(df[year_col]).astype("boolean")


def diff_gpa_term_1_to_year_1(
    df: pd.DataFrame,
    *,
    term_col: str = "gpa_group_term_1",
    year_col: str = "gpa_group_year_1",
) -> pd.Series:
    return df[year_col].sub(df[term_col])
