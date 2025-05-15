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
GPAField = ft.partial(pda.Field, nullable=True, ge=0.0, le=4.0)
NumCreditsGt0Field = ft.partial(pda.Field, nullable=True, ge=0.0)
GradeField = ft.partial(pda.Field, nullable=True)
CompletedGatewayField = ft.partial(
    pda.Field, nullable=True, dtype_kwargs={"categories": ["C", "D", "NA"]}
)
CompletedDevField = ft.partial(
    pda.Field, nullable=True, dtype_kwargs={"categories": ["Y", "N", "NA"]}
)
YearsToOfField = ft.partial(pda.Field, ge=0, le=8)


class RawPDPCohortDataSchema(pda.DataFrameModel):
    """
    Schema (aka ``DataFrameModel``) for raw PDP cohort data that validates columns,
    data types (including categorical categories), acceptable value ranges, and more.

    References:
        - https://help.studentclearinghouse.org/pdp/knowledge-base/cohort-level-analysis-ready-file-data-dictionary/
        - https://pandera.readthedocs.io/en/stable/dataframe_models.html
    """

    student_id: pt.Series["string"]
    institution_id: pt.Series["string"]
    cohort: pt.Series["string"]
    cohort_term: pt.Series[pd.CategoricalDtype] = TermField()
    enrollment_type: pt.Series[pd.CategoricalDtype]
    enrollment_intensity_first_term: pt.Series[pd.CategoricalDtype] = pda.Field(
        nullable=True
    )
    # NOTE: categories set in a parser, which forces "UK" values to null
    math_placement: pt.Series[pd.CategoricalDtype] = pda.Field(nullable=True)
    # NOTE: categories set in a parser, which forces "UK" values to null
    english_placement: pt.Series[pd.CategoricalDtype] = pda.Field(nullable=True)
    # NOTE: categories set in a parser, which forces "UK" values to null
    dual_and_summer_enrollment: pt.Series[pd.CategoricalDtype] = pda.Field(
        nullable=True
    )
    student_age: pt.Series[pd.StringDtype] = pda.Field(nullable=True)
    race: pt.Series[pd.StringDtype] = pda.Field(nullable=True)
    ethnicity: pt.Series[pd.StringDtype] = pda.Field(nullable=True)
    gender: pt.Series[pd.StringDtype] = pda.Field(nullable=True)
    first_gen: pt.Series[pd.StringDtype] = pda.Field(nullable=True)
    # NOTE: categories set in a parser, which forces "UK" values to null
    pell_status_first_year: pt.Series[pd.CategoricalDtype] = pda.Field(nullable=True)
    attendance_status_term_1: pt.Series[pd.StringDtype] = pda.Field(nullable=True)
    credential_type_sought_year_1: pt.Series[pd.StringDtype] = pda.Field(nullable=True)
    program_of_study_term_1: pt.Series["string"] = pda.Field(nullable=True)
    gpa_group_term_1: pt.Series["Float32"] = GPAField()
    gpa_group_year_1: pt.Series["Float32"] = GPAField()
    number_of_credits_attempted_year_1: pt.Series["Float32"] = pda.Field(
        nullable=True, ge=1.0, raise_warning=True
    )
    number_of_credits_earned_year_1: pt.Series["Float32"] = NumCreditsGt0Field()
    number_of_credits_attempted_year_2: pt.Series["Float32"] = NumCreditsGt0Field()
    number_of_credits_earned_year_2: pt.Series["Float32"] = NumCreditsGt0Field()
    number_of_credits_attempted_year_3: pt.Series["Float32"] = NumCreditsGt0Field()
    number_of_credits_earned_year_3: pt.Series["Float32"] = NumCreditsGt0Field()
    number_of_credits_attempted_year_4: pt.Series["Float32"] = NumCreditsGt0Field()
    number_of_credits_earned_year_4: pt.Series["Float32"] = NumCreditsGt0Field()
    # NOTE: categories set in a parser, which forces "UK" values to null
    gateway_math_status: pt.Series[pd.CategoricalDtype] = pda.Field(nullable=True)
    # NOTE: categories set in a parser, which forces "UK" values to null
    gateway_english_status: pt.Series[pd.CategoricalDtype] = pda.Field(nullable=True)
    # NOTE: categories set in a parser, which forces "UK" values to null
    attempted_gateway_math_year_1: pt.Series[pd.CategoricalDtype] = pda.Field(
        nullable=True
    )
    # NOTE: categories set in a parser, which forces "UK" values to null
    attempted_gateway_english_year_1: pt.Series[pd.CategoricalDtype] = pda.Field(
        nullable=True
    )
    completed_gateway_math_year_1: pt.Series[pd.CategoricalDtype] = (
        CompletedGatewayField()
    )
    completed_gateway_english_year_1: pt.Series[pd.CategoricalDtype] = (
        CompletedGatewayField()
    )
    gateway_math_grade_y_1: pt.Series["string"] = GradeField()
    gateway_english_grade_y_1: pt.Series["string"] = GradeField()
    # NOTE: categories set in a parser, which forces "UK" values to null
    attempted_dev_math_y_1: pt.Series[pd.CategoricalDtype] = pda.Field(nullable=True)
    # NOTE: categories set in a parser, which forces "UK" values to null
    attempted_dev_english_y_1: pt.Series[pd.CategoricalDtype] = pda.Field(nullable=True)
    completed_dev_math_y_1: pt.Series[pd.CategoricalDtype] = CompletedDevField()
    completed_dev_english_y_1: pt.Series[pd.CategoricalDtype] = CompletedDevField()
    retention: pt.Series["boolean"]
    persistence: pt.Series["boolean"]
    years_to_bachelors_at_cohort_inst: pt.Series["Int8"] = YearsToOfField(nullable=True)
    years_to_associates_or_certificate_at_cohort_inst: pt.Series["Int8"] = (
        YearsToOfField(nullable=True)
    )
    years_to_bachelor_at_other_inst: pt.Series["Int8"] = YearsToOfField(nullable=True)
    years_to_associates_or_certificate_at_other_inst: pt.Series["Int8"] = (
        YearsToOfField(nullable=True)
    )
    years_of_last_enrollment_at_cohort_institution: pt.Series["int32"] = (
        YearsToOfField()
    )
    years_of_last_enrollment_at_other_institution: pt.Series["int32"] = YearsToOfField()
    time_to_credential: pt.Series["Float32"] = pda.Field(nullable=True)
    # NOTE: categories set in a parser, which assigns "UK" values to null
    reading_placement: t.Optional[pt.Series[pd.CategoricalDtype]] = pda.Field(
        nullable=True
    )
    special_program: t.Optional[pt.Series["string"]] = pda.Field(nullable=True)
    naspa_first_generation: t.Optional[pt.Series[pd.CategoricalDtype]] = pda.Field(
        nullable=True,
        dtype_kwargs={"categories": ["-1", "0", "1", "2", "3", "4", "5", "6"]},
    )
    # NOTE: categories set in a parser, which assigns "UK" values to null
    incarcerated_status: t.Optional[pt.Series[pd.CategoricalDtype]] = pda.Field(
        nullable=True
    )
    # NOTE: categories set in a parser, which forces "-1" / "-1.0" values to null
    military_status: t.Optional[pt.Series[pd.CategoricalDtype]] = pda.Field(
        nullable=True
    )
    employment_status: t.Optional[pt.Series[pd.CategoricalDtype]] = pda.Field(
        nullable=True, dtype_kwargs={"categories": ["-1", "0", "1", "2", "3", "4"]}
    )
    # NOTE: categories set in a parser, which assigns "UK" values to null
    disability_status: t.Optional[pt.Series[pd.CategoricalDtype]] = pda.Field(
        nullable=True
    )
    foreign_language_completion: t.Optional[pt.Series["string"]] = pda.Field(
        nullable=True
    )
    first_year_to_bachelors_at_cohort_inst: pt.Series["Int8"] = YearsToOfField(
        nullable=True
    )
    first_year_to_associates_or_certificate_at_cohort_inst: pt.Series["Int8"] = (
        YearsToOfField(nullable=True)
    )
    first_year_to_bachelor_at_other_inst: pt.Series["Int8"] = YearsToOfField(
        nullable=True
    )
    first_year_to_associates_or_certificate_at_other_inst: pt.Series["Int8"] = (
        YearsToOfField(nullable=True)
    )
    program_of_study_year_1: pt.Series["string"] = pda.Field(nullable=True)
    most_recent_bachelors_at_other_institution_state: pt.Series["string"] = pda.Field(
        nullable=True
    )
    most_recent_associates_or_certificate_at_other_institution_state: pt.Series[
        "string"
    ] = pda.Field(nullable=True)
    most_recent_last_enrollment_at_other_institution_state: pt.Series["string"] = (
        pda.Field(nullable=True)
    )
    first_bachelors_at_other_institution_state: pt.Series["string"] = pda.Field(
        nullable=True
    )
    first_associates_or_certificate_at_other_institution_state: pt.Series["string"] = (
        pda.Field(nullable=True)
    )
    most_recent_bachelors_at_other_institution_carnegie: pt.Series["string"] = (
        pda.Field(nullable=True)
    )
    most_recent_associates_or_certificate_at_other_institution_carnegie: pt.Series[
        "string"
    ] = pda.Field(nullable=True)
    most_recent_last_enrollment_at_other_institution_carnegie: pt.Series["string"] = (
        pda.Field(nullable=True)
    )
    first_bachelors_at_other_institution_carnegie: pt.Series["string"] = pda.Field(
        nullable=True
    )
    first_associates_or_certificate_at_other_institution_carnegie: pt.Series[
        "string"
    ] = pda.Field(nullable=True)
    most_recent_bachelors_at_other_institution_locale: pt.Series[pd.StringDtype] = (
        pda.Field(nullable=True)
    )
    most_recent_associates_or_certificate_at_other_institution_locale: pt.Series[
        pd.StringDtype
    ] = pda.Field(nullable=True)
    most_recent_last_enrollment_at_other_institution_locale: pt.Series[
        pd.StringDtype
    ] = pda.Field(nullable=True)
    first_bachelors_at_other_institution_locale: pt.Series[pd.StringDtype] = pda.Field(
        nullable=True
    )
    first_associates_or_certificate_at_other_institution_locale: pt.Series[
        pd.StringDtype
    ] = pda.Field(nullable=True)
    # added in 2025-01
    years_to_latest_associates_at_cohort_inst: t.Optional[pt.Series["Int8"]] = (
        YearsToOfField(nullable=True)
    )
    years_to_latest_certificate_at_cohort_inst: t.Optional[pt.Series["Int8"]] = (
        YearsToOfField(nullable=True)
    )
    years_to_latest_associates_at_other_inst: t.Optional[pt.Series["Int8"]] = (
        YearsToOfField(nullable=True)
    )
    years_to_latest_certificate_at_other_inst: t.Optional[pt.Series["Int8"]] = (
        YearsToOfField(nullable=True)
    )
    first_year_to_associates_at_cohort_inst: t.Optional[pt.Series["Int8"]] = (
        YearsToOfField(nullable=True)
    )
    first_year_to_certificate_at_cohort_inst: t.Optional[pt.Series["Int8"]] = (
        YearsToOfField(nullable=True)
    )
    first_year_to_associates_at_other_inst: t.Optional[pt.Series["Int8"]] = (
        YearsToOfField(nullable=True)
    )
    first_year_to_certificate_at_other_inst: t.Optional[pt.Series["Int8"]] = (
        YearsToOfField(nullable=True)
    )

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
        "first_gen",
        "credential_type_sought_year_1",
        "most_recent_bachelors_at_other_institution_locale",
        "most_recent_associates_or_certificate_at_other_institution_locale",
        "most_recent_last_enrollment_at_other_institution_locale",
        "first_bachelors_at_other_institution_locale",
        "first_associates_or_certificate_at_other_institution_locale",
    )
    def strip_and_uppercase_strings(cls, series):
        return series.str.strip().str.upper()

    @pda.parser("enrollment_type")
    def set_enrollment_type_categories(cls, series):
        return _strip_upper_strings_to_cats(series).cat.set_categories(
            ["FIRST-TIME", "RE-ADMIT", "TRANSFER-IN"]
        )

    @pda.parser("enrollment_intensity_first_term")
    def set_enrollment_intensity_first_term_categories(cls, series):
        return _strip_upper_strings_to_cats(series).cat.set_categories(
            ["FULL-TIME", "PART-TIME"]
        )

    @pda.parser("math_placement", "english_placement", "reading_placement")
    def set_subj_placement_categories(cls, series):
        return _strip_upper_strings_to_cats(series).cat.set_categories(["C", "N"])

    @pda.parser("dual_and_summer_enrollment")
    def set_dual_and_summer_enrollment_categories(cls, series):
        return _strip_upper_strings_to_cats(series).cat.set_categories(
            ["DE", "SE", "DS"]
        )

    @pda.parser("pell_status_first_year")
    def set_pell_status_first_year_categories(cls, series):
        return _strip_upper_strings_to_cats(series).cat.set_categories(["Y", "N"])

    @pda.parser("gateway_math_status", "gateway_english_status")
    def set_gateway_math_english_status_categories(cls, series):
        return _strip_upper_strings_to_cats(series).cat.set_categories(["R", "N"])

    @pda.parser("attempted_gateway_math_year_1", "attempted_gateway_english_year_1")
    def set_attempted_gateway_math_english_year_1_categories(cls, series):
        return _strip_upper_strings_to_cats(series).cat.set_categories(["Y", "N"])

    @pda.parser("attempted_dev_math_y_1", "attempted_dev_english_y_1")
    def set_attempted_dev_math_english_y_1_categories(cls, series):
        return _strip_upper_strings_to_cats(series).cat.set_categories(["Y", "N", "NA"])

    @pda.parser("incarcerated_status")
    def set_incarcerated_status_categories(cls, series):
        return _strip_upper_strings_to_cats(series).cat.set_categories(["Y", "P", "N"])

    @pda.parser("disability_status")
    def set_disability_status_categories(cls, series):
        return _strip_upper_strings_to_cats(series).cat.set_categories(["Y", "N"])

    @pda.parser("military_status")
    def set_military_status_categories(cls, series):
        return (
            series.astype("Float32")
            .astype("Int8")
            .astype("category")
            .cat.set_categories(["0", "1", "2"])
        )

    @pda.parser(
        "years_to_associates_or_certificate_at_cohort_inst",
        "years_to_bachelors_at_cohort_inst",
        "years_to_associates_or_certificate_at_other_inst",
        "years_to_bachelor_at_other_inst",
        "first_year_to_associates_or_certificate_at_cohort_inst",
        "first_year_to_bachelors_at_cohort_inst",
        "first_year_to_associates_or_certificate_at_other_inst",
        "first_year_to_bachelor_at_other_inst",
        "years_to_latest_associates_at_cohort_inst",
        "years_to_latest_certificate_at_cohort_inst",
        "years_to_latest_associates_at_other_inst",
        "years_to_latest_certificate_at_other_inst",
        "first_year_to_associates_at_cohort_inst",
        "first_year_to_certificate_at_cohort_inst",
        "first_year_to_associates_at_other_inst",
        "first_year_to_certificate_at_other_inst",
    )
    def set_zero_year_values_to_null(cls, series):
        return series.mask(series.eq(0), pd.NA).astype("Int8")

    @pda.check("institution_id", name="check_num_institutions")
    def check_num_institutions(cls, series) -> bool:
        return series.nunique() == 1

    class Config:
        coerce = True
        # "strict" parsing is disabled so we can rename raw identifier cols to student_id
        strict = False
        unique_column_names = True
        add_missing_columns = False
        drop_invalid_rows = False
        unique = ["student_id"]


def _strip_upper_strings_to_cats(series: pd.Series) -> pd.Series:
    return series.str.strip().str.upper().astype("category")
