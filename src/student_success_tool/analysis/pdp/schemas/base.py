# ruff: noqa: F821
# mypy: ignore-errors
import functools as ft
import typing as t

import pandas as pd
import pandera as pda
import pandera.typing as pt

StudentAgeField = ft.partial(
    pda.Field,
    dtype_kwargs={
        "categories": ["20 AND YOUNGER", ">20 - 24", "OLDER THAN 24"],
        "ordered": True,
    },
)
RaceField = ft.partial(
    pda.Field,
    dtype_kwargs={
        "categories": [
            "NONRESIDENT ALIEN",
            "AMERICAN INDIAN OR ALASKA NATIVE",
            "ASIAN",
            "BLACK OR AFRICAN AMERICAN",
            "NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER",
            "WHITE",
            "HISPANIC",
            "TWO OR MORE RACES",
            "UNKNOWN",
        ],
    },
)
EthnicityField = ft.partial(pda.Field, dtype_kwargs={"categories": ["H", "N", "UK"]})
GenderField = ft.partial(
    pda.Field, dtype_kwargs={"categories": ["M", "F", "P", "X", "UK"]}
)
TermField = ft.partial(
    pda.Field,
    dtype_kwargs={
        "categories": ["FALL", "WINTER", "SPRING", "SUMMER"],
        "ordered": True,
    },
)
GPAField = ft.partial(pda.Field, nullable=True, ge=0.0, le=4.0)
NumCreditsGt0Field = ft.partial(pda.Field, nullable=True, ge=0.0)
GradeField = ft.partial(
    pda.Field,
    nullable=True,
    dtype_kwargs={
        "categories": ["0", "1", "2", "3", "4", "P", "F", "I", "W", "A", "M", "O"]
    },
)
CompletedDevOrGatewayField = ft.partial(
    pda.Field, nullable=True, dtype_kwargs={"categories": ["C", "D", "NA"]}
)
YearsToOfField = ft.partial(pda.Field, ge=0, le=7)
LocaleField = ft.partial(
    pda.Field,
    nullable=True,
    dtype_kwargs={"categories": ["URBAN", "SUBURB", "TOWN/RURAL"]},
)


class RawPDPCourseDataSchema(pda.DataFrameModel):
    """
    Schema (aka ``DataFrameModel``) for raw PDP course data that validates columns,
    data types (including categorical categories), acceptable value ranges, and more.

    References:
        - https://help.studentclearinghouse.org/pdp/knowledge-base/course-level-analysis-ready-file-data-dictionary
        - https://pandera.readthedocs.io/en/stable/dataframe_models.html
    """

    student_guid: pt.Series["string"]
    institution_id: pt.Series["string"]
    student_age: pt.Series[pd.CategoricalDtype] = StudentAgeField()
    race: pt.Series[pd.CategoricalDtype] = RaceField()
    ethnicity: pt.Series[pd.CategoricalDtype] = EthnicityField()
    gender: pt.Series[pd.CategoricalDtype] = GenderField()
    cohort: pt.Series["string"]
    cohort_term: pt.Series[pd.CategoricalDtype] = TermField()
    academic_year: pt.Series["string"]
    academic_term: pt.Series[pd.CategoricalDtype] = TermField()
    course_prefix: pt.Series["string"]
    course_number: pt.Series["string"]
    section_id: pt.Series["string"]
    course_name: pt.Series["string"]
    course_cip: pt.Series["string"]
    course_type: pt.Series[pd.CategoricalDtype] = pda.Field(
        dtype_kwargs={
            "categories": ["CU", "CG", "CC", "CD", "EL", "AB", "GE", "NC", "O"]
        },
    )
    math_or_english_gateway: pt.Series[pd.CategoricalDtype] = pda.Field(
        nullable=True, dtype_kwargs={"categories": ["M", "E", "NA"]}
    )
    co_requisite_course: pt.Series[pd.CategoricalDtype] = pda.Field(
        dtype_kwargs={"categories": ["Y", "N"]}
    )
    course_begin_date: pt.Series["datetime64[ns]"]
    course_end_date: pt.Series["datetime64[ns]"]
    grade: pt.Series[pd.CategoricalDtype] = GradeField()
    number_of_credits_attempted: pt.Series["Float32"] = NumCreditsGt0Field(le=20)
    number_of_credits_earned: pt.Series["Float32"] = NumCreditsGt0Field(le=20)
    delivery_method: pt.Series[pd.CategoricalDtype] = pda.Field(
        nullable=True, dtype_kwargs={"categories": ["F", "O", "H"]}
    )
    core_course: pt.Series[pd.CategoricalDtype] = pda.Field(
        nullable=True, dtype_kwargs={"categories": ["Y", "N"]}
    )
    core_course_type: pt.Series["string"] = pda.Field(nullable=True)
    core_competency_completed: pt.Series[pd.CategoricalDtype] = pda.Field(
        nullable=True, dtype_kwargs={"categories": ["Y", "N"]}
    )
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
        nullable=True, dtype_kwargs={"categories": ["1", "2", "3", "4", "5", "6", "7"]}
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
        strict = True
        unique_column_names = True
        add_missing_columns = False
        drop_invalid_rows = False
        unique = [
            "student_guid",
            "academic_year",
            "academic_term",
            "course_prefix",
            "course_number",
            "section_id",
        ]


class RawPDPCohortDataSchema(pda.DataFrameModel):
    """
    Schema (aka ``DataFrameModel``) for raw PDP cohort data that validates columns,
    data types (including categorical categories), acceptable value ranges, and more.

    References:
        - https://help.studentclearinghouse.org/pdp/knowledge-base/cohort-level-analysis-ready-file-data-dictionary/
        - https://pandera.readthedocs.io/en/stable/dataframe_models.html
    """

    student_guid: pt.Series["string"]
    institution_id: pt.Series["string"]
    cohort: pt.Series["string"]
    cohort_term: pt.Series[pd.CategoricalDtype] = TermField()
    student_age: pt.Series[pd.CategoricalDtype] = StudentAgeField()
    enrollment_type: pt.Series[pd.CategoricalDtype] = pda.Field(
        dtype_kwargs={"categories": ["FIRST-TIME", "RE-ADMIT", "TRANSFER-IN"]},
    )
    enrollment_intensity_first_term: pt.Series[pd.CategoricalDtype] = pda.Field(
        dtype_kwargs={"categories": ["FULL-TIME", "PART-TIME"]},
    )
    # NOTE: categories set in a parser, which forces "UK" values to null
    math_placement: pt.Series[pd.CategoricalDtype] = pda.Field(nullable=True)
    # NOTE: categories set in a parser, which forces "UK" values to null
    english_placement: pt.Series[pd.CategoricalDtype] = pda.Field(nullable=True)
    # NOTE: categories set in a parser, which forces "UK" values to null
    dual_and_summer_enrollment: pt.Series[pd.CategoricalDtype] = pda.Field(
        nullable=True
    )
    race: pt.Series[pd.CategoricalDtype] = RaceField()
    ethnicity: pt.Series[pd.CategoricalDtype] = EthnicityField()
    gender: pt.Series[pd.CategoricalDtype] = GenderField()
    first_gen: pt.Series[pd.CategoricalDtype] = pda.Field(
        nullable=True, dtype_kwargs={"categories": ["P", "C", "A", "B"]}
    )
    # NOTE: categories set in a parser, which forces "UK" values to null
    pell_status_first_year: pt.Series[pd.CategoricalDtype] = pda.Field(nullable=True)
    attendance_status_term_1: pt.Series[pd.CategoricalDtype] = pda.Field(
        dtype_kwargs={
            "categories": [
                "First-Time Full-Time",
                "First-Time Part-Time",
                "First-Time Unknown",
                "Transfer-In Full-Time",
                "Transfer-In Part-Time",
                "Transfer-In Unknown",
                "Re-admit Full-Time",
                "Re-admit Part-Time",
                "Re-admit Unknown",
            ]
        },
    )
    credential_type_sought_year_1: pt.Series[pd.CategoricalDtype] = pda.Field(
        dtype_kwargs={
            "categories": [
                "Less than one-year certificate, less than Associate degree",
                "One to two year certificate, less than Associate degree",
                "Two to four year certificate, less than Bachelor's degree",
                "Undergraduate Certificate or Diploma Program",
                "Associate Degree",
                "Bachelor's Degree",
                "Post Baccalaureate Certificate",
                "Master's Degree",
                "Doctoral Degree",
                "First Professional Degree",
                "Graduate/Professional Certificate",
                "Non- Credential Program (Preparatory Coursework/Teach Certification)",
                "Missing",
            ]
        },
    )
    program_of_study_term_1: pt.Series["string"] = pda.Field(nullable=True)
    gpa_group_term_1: pt.Series["Float32"] = GPAField()
    gpa_group_year_1: pt.Series["Float32"] = GPAField()
    # this is the only credits field required to be >= 1
    number_of_credits_attempted_year_1: pt.Series["Float32"] = pda.Field(
        nullable=True, ge=1.0
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
        CompletedDevOrGatewayField()
    )
    completed_gateway_english_year_1: pt.Series[pd.CategoricalDtype] = (
        CompletedDevOrGatewayField()
    )
    gateway_math_grade_y_1: pt.Series[pd.CategoricalDtype] = GradeField()
    gateway_english_grade_y_1: pt.Series[pd.CategoricalDtype] = GradeField()
    # NOTE: categories set in a parser, which forces "UK" values to null
    attempted_dev_math_y_1: pt.Series[pd.CategoricalDtype] = pda.Field(nullable=True)
    # NOTE: categories set in a parser, which forces "UK" values to null
    attempted_dev_english_y_1: pt.Series[pd.CategoricalDtype] = pda.Field(nullable=True)
    completed_dev_math_y_1: pt.Series[pd.CategoricalDtype] = (
        CompletedDevOrGatewayField()
    )
    completed_dev_english_y_1: pt.Series[pd.CategoricalDtype] = (
        CompletedDevOrGatewayField()
    )
    retention: pt.Series["boolean"]
    persistence: pt.Series["boolean"]
    years_to_bachelors_at_cohort_inst: pt.Series["int32"] = YearsToOfField()
    years_to_associates_or_certificate_at_cohort_inst: pt.Series["int32"] = (
        YearsToOfField()
    )
    years_to_bachelor_at_other_inst: pt.Series["int32"] = YearsToOfField()
    years_to_associates_or_certificate_at_other_inst: pt.Series["int32"] = (
        YearsToOfField()
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
    military_status: t.Optional[pt.Series[pd.CategoricalDtype]] = pda.Field(
        nullable=True, dtype_kwargs={"categories": ["-1", "0", "1", "2"]}
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
    most_recent_bachelors_at_other_institution_locale: pt.Series[
        pd.CategoricalDtype
    ] = LocaleField()
    most_recent_associates_or_certificate_at_other_institution_locale: pt.Series[
        pd.CategoricalDtype
    ] = LocaleField()
    most_recent_last_enrollment_at_other_institution_locale: pt.Series[
        pd.CategoricalDtype
    ] = LocaleField()
    first_bachelors_at_other_institution_locale: pt.Series[pd.CategoricalDtype] = (
        LocaleField()
    )
    first_associates_or_certificate_at_other_institution_locale: pt.Series[
        pd.CategoricalDtype
    ] = LocaleField()

    @pda.parser("math_placement", "english_placement", "reading_placement")
    def set_subj_placement_categories(cls, series):
        return series.cat.set_categories(["C", "N"])

    @pda.parser("dual_and_summer_enrollment")
    def set_dual_and_summer_enrollment_categories(cls, series):
        return series.cat.set_categories(["DE", "SE", "DS"])

    @pda.parser("pell_status_first_year")
    def set_pell_status_first_year_categories(cls, series):
        return series.cat.set_categories(["Y", "N"])

    @pda.parser("gateway_math_status", "gateway_english_status")
    def set_gateway_math_english_status_categories(cls, series):
        return series.cat.set_categories(["R", "N"])

    @pda.parser("attempted_gateway_math_year_1", "attempted_gateway_english_year_1")
    def set_attempted_gateway_math_english_year_1_categories(cls, series):
        return series.cat.set_categories(["Y", "N"])

    @pda.parser("attempted_dev_math_y_1", "attempted_dev_english_y_1")
    def set_attempted_dev_math_english_y_1_categories(cls, series):
        return series.cat.set_categories(["Y", "N", "NA"])

    @pda.parser("incarcerated_status")
    def set_incarcerated_status_categories(cls, series):
        return series.cat.set_categories(["Y", "P", "N"])

    @pda.parser("disability_status")
    def set_disability_status_categories(cls, series):
        return series.cat.set_categories(["Y", "N"])

    @pda.parser(
        "first_year_to_associates_or_certificate_at_cohort_inst",
        "first_year_to_bachelors_at_cohort_inst",
        "first_year_to_associates_or_certificate_at_other_inst",
        "first_year_to_bachelor_at_other_inst",
    )
    def set_zero_year_values_to_null(cls, series):
        return series.mask(series.eq(0), pd.NA).astype("Int8")

    class Config:
        coerce = True
        strict = True
        unique_column_names = True
        add_missing_columns = False
        drop_invalid_rows = False
        unique = ["institution_id", "student_guid"]
