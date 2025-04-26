# ruff: noqa: F821
# mypy: ignore-errors
import functools as ft
import logging
import typing as t

import pandas as pd

try:
    import pandera as pda
    import pandera.typing as pt
except ModuleNotFoundError:
    from ... import utils

    utils.databricks.mock_pandera()

    import pandera as pda
    import pandera.typing as pt

LOGGER = logging.getLogger(__name__)

TermField = ft.partial(
    pda.Field,
    dtype_kwargs={
        "categories": ["FALL", "WINTER", "SPRING", "SUMMER"],
        "ordered": True,
    },
)
NumCreditsGt0Field = ft.partial(pda.Field, nullable=True, ge=0.0)


class RawPDPCourseDataSchema(pda.DataFrameModel):
    """
    Schema (aka ``DataFrameModel``) for raw PDP course data that validates columns,
    data types (including categorical categories), acceptable value ranges, and more.

    References:
        - https://help.studentclearinghouse.org/pdp/knowledge-base/course-level-analysis-ready-file-data-dictionary
        - https://pandera.readthedocs.io/en/stable/dataframe_models.html
    """

    student_id: pt.Series["string"]
    institution_id: pt.Series["string"]
    student_age: pt.Series[pd.StringDtype] = pda.Field(nullable=True)
    race: pt.Series[pd.StringDtype] = pda.Field(nullable=True)
    ethnicity: pt.Series[pd.StringDtype] = pda.Field(nullable=True)
    gender: pt.Series[pd.StringDtype] = pda.Field(nullable=True)
    cohort: pt.Series["string"]
    cohort_term: pt.Series[pd.CategoricalDtype] = TermField()
    academic_year: pt.Series["string"] = pda.Field(nullable=True)
    academic_term: pt.Series[pd.CategoricalDtype] = TermField(nullable=True)
    course_prefix: pt.Series["string"] = pda.Field(nullable=True)
    course_number: pt.Series["string"] = pda.Field(nullable=True)
    section_id: pt.Series["string"] = pda.Field(nullable=True)
    course_name: pt.Series["string"] = pda.Field(nullable=True)
    course_cip: pt.Series["string"] = pda.Field(nullable=True)
    course_type: pt.Series[pd.CategoricalDtype] = pda.Field(
        nullable=True,
        dtype_kwargs={
            "categories": ["CU", "CG", "CC", "CD", "EL", "AB", "GE", "NC", "O"]
        },
    )
    math_or_english_gateway: pt.Series[pd.CategoricalDtype] = pda.Field(nullable=True)
    co_requisite_course: pt.Series[pd.CategoricalDtype] = pda.Field(
        nullable=True, dtype_kwargs={"categories": ["Y", "N"]}
    )
    course_begin_date: pt.Series["datetime64[ns]"] = pda.Field(nullable=True)
    course_end_date: pt.Series["datetime64[ns]"] = pda.Field(nullable=True)
    grade: pt.Series["string"] = pda.Field(nullable=True)
    number_of_credits_attempted: pt.Series["Float32"] = NumCreditsGt0Field(le=20)
    number_of_credits_earned: pt.Series["Float32"] = NumCreditsGt0Field(le=20)
    delivery_method: pt.Series[pd.CategoricalDtype] = pda.Field(
        nullable=True, dtype_kwargs={"categories": ["F", "O", "H"]}
    )
    core_course: pt.Series[pd.CategoricalDtype] = pda.Field(
        nullable=True, dtype_kwargs={"categories": ["Y", "N"]}
    )
    core_course_type: pt.Series["string"] = pda.Field(nullable=True)
    core_competency_completed: pt.Series[pd.CategoricalDtype] = pda.Field(nullable=True)
    enrolled_at_other_institution_s: pt.Series[pd.CategoricalDtype] = pda.Field(
        nullable=True, dtype_kwargs={"categories": ["Y", "N"]}
    )
    credential_engine_identifier: t.Optional[pt.Series["string"]] = pda.Field(
        nullable=True
    )
    course_instructor_employment_status: t.Optional[pt.Series[pd.CategoricalDtype]] = (
        pda.Field(nullable=True, dtype_kwargs={"categories": ["PT", "FT"]})
    )
    course_instructor_rank: t.Optional[pt.Series[pd.CategoricalDtype]] = pda.Field(
        nullable=True
    )
    enrollment_record_at_other_institution_s_state_s: pt.Series["string"] = pda.Field(
        nullable=True,
    )
    enrollment_record_at_other_institution_s_carnegie_s: pt.Series["string"] = (
        pda.Field(nullable=True)
    )
    enrollment_record_at_other_institution_s_locale_s: pt.Series["string"] = pda.Field(
        nullable=True
    )
    # added in 2025-01
    term_program_of_study: t.Optional[pt.Series["string"]] = pda.Field(nullable=True)

    @pda.dataframe_parser
    def rename_student_id_col(cls, df):
        if "study_id" in df.columns:
            LOGGER.info("renaming 'study_id' column => 'student_id'")
            df = df.rename(columns={"study_id": "student_id"}).astype(
                {"student_id": "string"}
            )
        elif "student_guid" in df.columns:
            LOGGER.info("renaming 'student_guid' column => 'student_id'")
            df = df.rename(columns={"student_guid": "student_id"}).astype(
                {"student_id": "string"}
            )
        return df

    @pda.parser(
        "student_age",
        "race",
        "ethnicity",
        "gender",
        "enrollment_record_at_other_institution_s_state_s",
        "enrollment_record_at_other_institution_s_locale_s",
    )
    def strip_and_uppercase_strings(cls, series):
        return series.str.strip().str.upper()

    @pda.parser("math_or_english_gateway")
    def set_math_or_english_gateway_categories(cls, series):
        return _strip_upper_strings_to_cats(series).cat.set_categories(["E", "M", "NA"])

    @pda.parser("core_competency_completed")
    def set_core_competency_completed_categories(cls, series):
        return _strip_upper_strings_to_cats(series).cat.set_categories(["Y", "N"])

    @pda.parser("course_instructor_rank")
    def set_course_instructor_rank_categories(cls, series):
        return series.cat.set_categories(["1", "2", "3", "4", "5", "6", "7"])

    @pda.dataframe_check
    def num_credits_attempted_ge_earned(cls, df: pd.DataFrame) -> pd.Series:
        col_attempted = "number_of_credits_attempted"
        col_earned = "number_of_credits_earned"
        return (
            df[col_attempted].ge(df[col_earned])
            # since pandas treats NA != NA, we also need to allow for nulls
            | df[[col_attempted, col_earned]].isna().any(axis="columns")
        )

    class Config:
        coerce = True
        # "strict" parsing is disabled so we can rename raw identifier cols to student_id
        strict = False
        unique_column_names = True
        add_missing_columns = False
        drop_invalid_rows = False
        unique = [
            "student_id",
            "academic_year",
            "academic_term",
            "course_prefix",
            "course_number",
            "section_id",
        ]


def _strip_upper_strings_to_cats(series: pd.Series) -> pd.Series:
    return series.str.strip().str.upper().astype("category")
