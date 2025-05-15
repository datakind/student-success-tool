import functools as ft
import logging

import pandas as pd

from ....utils import types
from . import constants, shared

LOGGER = logging.getLogger(__name__)


def add_features(
    df: pd.DataFrame,
    *,
    first_term_of_year: types.TermType = constants.DEFAULT_FIRST_TERM_OF_YEAR,  # type: ignore
) -> pd.DataFrame:
    """
    Compute student-level features from a standardized cohort dataset,
    and add as columns to ``df`` .

    Args:
        df
        first_term_of_year
    """
    LOGGER.info("adding student features ...")
    # in case somebody drops some/all years, check for which years are present in df
    credits_years = [
        yr for yr in (1, 2, 3, 4) if f"number_of_credits_earned_year_{yr}" in df.columns
    ]
    return df.assign(
        cohort_id=ft.partial(
            shared.year_term, year_col="cohort", term_col="cohort_term"
        ),
        cohort_start_dt=ft.partial(
            shared.year_term_dt,
            col="cohort_id",
            bound="start",
            first_term_of_year=first_term_of_year,
        ),
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
        student_is_pell_recipient_first_year=student_is_pell_recipient_first_year,
        diff_gpa_term_1_to_year_1=diff_gpa_term_1_to_year_1,
        **{
            f"frac_credits_earned_year_{yr}": ft.partial(
                shared.frac_credits_earned,
                earned_col=f"number_of_credits_earned_year_{yr}",
                attempted_col=f"number_of_credits_attempted_year_{yr}",
            )
            for yr in credits_years
        },
    )


def student_program_of_study_area(df: pd.DataFrame, *, col: str) -> pd.Series:
    return shared.extract_short_cip_code(df[col])


def student_program_of_study_changed_term_1_to_year_1(
    df: pd.DataFrame, *, term_col: str, year_col: str
) -> pd.Series:
    return df[term_col].ne(df[year_col]).astype("boolean")


def student_is_pell_recipient_first_year(
    df: pd.DataFrame,
    *,
    pell_col: str = "pell_status_first_year",
) -> pd.Series:
    return df[pell_col].map({"Y": True, "N": False}).astype("boolean")


def diff_gpa_term_1_to_year_1(
    df: pd.DataFrame,
    *,
    term_col: str = "gpa_group_term_1",
    year_col: str = "gpa_group_year_1",
) -> pd.Series:
    return df[year_col].sub(df[term_col])
