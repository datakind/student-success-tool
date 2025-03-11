# ruff: noqa: F821
# mypy: ignore-errors
import functools as ft
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

# TODO: re-use existing fields from raw data schemas?

TermField = ft.partial(
    pda.Field,
    dtype_kwargs={
        "categories": ["FALL", "WINTER", "SPRING", "SUMMER"],
        # TODO: do these need to be ordered anymore?
        # if so, how should we handle schools with alt orderings?
        # "ordered": True,
    },
)
PlacementField = ft.partial(
    pda.Field, nullable=True, dtype_kwargs={"categories": ["C", "N"]}
)
GPAField = ft.partial(pda.Field, nullable=True, ge=0.0, le=4.0)
NumCreditsGt0Field = ft.partial(pda.Field, nullable=True, ge=0.0)


class PDPStudentTermsDataSchema(pda.DataFrameModel):
    student_id: pt.Series["string"]
    term_id: pt.Series["string"]
    institution_id: pt.Series["string"]
    academic_year: pt.Series["string"]
    academic_term: pt.Series[pd.CategoricalDtype] = TermField()
    term_start_dt: pt.Series["datetime64[ns]"]
    term_rank: pt.Series["int8"]
    term_in_peak_covid: pt.Series["bool"]
    term_rank_core: pt.Series["Int8"] = pda.Field(nullable=True)
    term_is_core: pt.Series["bool"]
    num_courses: pt.Series["int8"]
    num_courses_passed: pt.Series["int8"]
    num_courses_completed: pt.Series["int8"]
    num_credits_attempted: pt.Series["float32"]
    num_credits_earned: pt.Series["float32"]
    course_ids: pt.Series["object"]
    course_subjects: pt.Series["object"]
    course_subject_areas: pt.Series["object"]
    course_id_nunique: pt.Series["Int8"] = pda.Field(nullable=True)
    course_subject_nunique: pt.Series["Int8"] = pda.Field(nullable=True)
    course_subject_area_nunique: pt.Series["Int8"] = pda.Field(nullable=True)
    course_level_mean: pt.Series["Float32"] = pda.Field(nullable=True)
    course_level_std: pt.Series["Float32"] = pda.Field(nullable=True)
    course_grade_numeric_mean: pt.Series["Float32"] = pda.Field(nullable=True)
    course_grade_numeric_std: pt.Series["Float32"] = pda.Field(nullable=True)
    section_num_students_enrolled_mean: pt.Series["Float32"] = pda.Field(nullable=True)
    section_num_students_enrolled_std: pt.Series["Float32"] = pda.Field(nullable=True)
    sections_num_students_enrolled: pt.Series["int8"]
    sections_num_students_passed: pt.Series["int8"]
    sections_num_students_completed: pt.Series["int8"]
    num_courses_enrolled_at_other_institution_s_y: pt.Series["int8"]
    num_courses_core_course_y: t.Optional[pt.Series["int8"]]
    num_courses_core_course_n: t.Optional[pt.Series["int8"]]
    num_courses_course_type_cc_cd: pt.Series["int8"]
    num_courses_course_type_cu: pt.Series["int8"]
    num_courses_course_type_cg: pt.Series["int8"]
    num_courses_course_type_cc: pt.Series["int8"]
    num_courses_course_type_cd: pt.Series["int8"]
    num_courses_course_type_el: pt.Series["int8"]
    num_courses_course_type_ab: pt.Series["int8"]
    num_courses_course_type_ge: pt.Series["int8"]
    num_courses_course_type_nc: pt.Series["int8"]
    num_courses_course_type_o: pt.Series["int8"]
    num_courses_delivery_method_f: pt.Series["int8"]
    num_courses_delivery_method_o: pt.Series["int8"]
    num_courses_delivery_method_h: pt.Series["int8"]
    num_courses_math_or_english_gateway_m: pt.Series["int8"]
    num_courses_math_or_english_gateway_e: pt.Series["int8"]
    num_courses_math_or_english_gateway_na: pt.Series["int8"]
    num_courses_co_requisite_course_y: pt.Series["int8"]
    num_courses_co_requisite_course_n: pt.Series["int8"]
    num_courses_course_instructor_employment_status_pt: pt.Series["int8"]
    num_courses_course_instructor_employment_status_ft: pt.Series["int8"]
    num_courses_course_instructor_rank_1: pt.Series["int8"]
    num_courses_course_instructor_rank_2: pt.Series["int8"]
    num_courses_course_instructor_rank_3: pt.Series["int8"]
    num_courses_course_instructor_rank_4: pt.Series["int8"]
    num_courses_course_instructor_rank_5: pt.Series["int8"]
    num_courses_course_instructor_rank_6: pt.Series["int8"]
    num_courses_course_instructor_rank_7: pt.Series["int8"]
    # NOTE: let's cover most likely values, not all possible values
    num_courses_course_level_1: pt.Series["int8"]
    num_courses_course_grade_a: pt.Series["int8"]
    num_courses_grade_is_failing_or_withdrawal: pt.Series["int8"]
    num_courses_grade_above_section_avg: pt.Series["int8"]
    cohort: pt.Series["string"]
    cohort_term: pt.Series[pd.CategoricalDtype] = TermField()
    # TODO: re-use existing fields from raw data schemas?
    enrollment_type: pt.Series[pd.CategoricalDtype] = pda.Field(
        nullable=True,
        dtype_kwargs={"categories": ["FIRST-TIME", "RE-ADMIT", "TRANSFER-IN"]},
    )
    enrollment_intensity_first_term: pt.Series[pd.CategoricalDtype] = pda.Field(
        nullable=True,
        dtype_kwargs={"categories": ["FULL-TIME", "PART-TIME"]},
    )
    math_placement: pt.Series[pd.CategoricalDtype] = PlacementField()
    english_placement: pt.Series[pd.CategoricalDtype] = PlacementField()
    reading_placement: pt.Series[pd.CategoricalDtype] = PlacementField()
    dual_and_summer_enrollment: pt.Series[pd.CategoricalDtype] = pda.Field(
        nullable=True,
        dtype_kwargs={"categories": ["DE", "SE", "DS"]},
    )
    # NOTE: intentionally not bothering with categoricals for student demographics
    student_age: pt.Series["string"]
    race: pt.Series["string"]
    ethnicity: pt.Series["string"]
    first_gen: pt.Series["string"]
    pell_status_first_year: pt.Series["string"] = pda.Field(nullable=True)
    special_program: t.Optional[pt.Series["string"]] = pda.Field(nullable=True)
    naspa_first_generation: t.Optional[pt.Series["string"]] = pda.Field(nullable=True)
    incarcerated_status: t.Optional[pt.Series["string"]] = pda.Field(nullable=True)
    military_status: t.Optional[pt.Series["string"]] = pda.Field(nullable=True)
    employment_status: t.Optional[pt.Series["string"]] = pda.Field(nullable=True)
    disability_status: t.Optional[pt.Series["string"]] = pda.Field(nullable=True)
    foreign_language_completion: t.Optional[pt.Series["string"]] = pda.Field(
        nullable=True
    )
    credential_type_sought_year_1: pt.Series["string"] = pda.Field(nullable=True)
    program_of_study_term_1: pt.Series["string"] = pda.Field(nullable=True)
    program_of_study_year_1: pt.Series["string"] = pda.Field(nullable=True)
    gpa_group_term_1: pt.Series["Float32"] = GPAField()
    gpa_group_year_1: pt.Series["Float32"] = GPAField()
    number_of_credits_earned_year_1: pt.Series["Float32"] = NumCreditsGt0Field()
    number_of_credits_attempted_year_2: pt.Series["Float32"] = NumCreditsGt0Field()
    number_of_credits_earned_year_2: pt.Series["Float32"] = NumCreditsGt0Field()
    number_of_credits_attempted_year_3: pt.Series["Float32"] = NumCreditsGt0Field()
    number_of_credits_earned_year_3: pt.Series["Float32"] = NumCreditsGt0Field()
    number_of_credits_attempted_year_4: pt.Series["Float32"] = NumCreditsGt0Field()
    number_of_credits_earned_year_4: pt.Series["Float32"] = NumCreditsGt0Field()
    # NOTE: let's skip the many "outcome"-like columns here; we'll drop them later, anyway
    cohort_id: pt.Series["string"]
    cohort_start_dt: pt.Series["datetime64[ns]"]
    student_program_of_study_area_term_1: pt.Series["string"] = pda.Field(nullable=True)
    student_program_of_study_area_year_1: pt.Series["string"] = pda.Field(nullable=True)
    student_program_of_study_area_changed_term_1_to_year_1: pt.Series["boolean"] = (
        pda.Field(nullable=True)
    )
    diff_gpa_term_1_to_year_1: pt.Series["Float32"] = pda.Field(nullable=True)
    frac_credits_earned_year_1: pt.Series["Float32"] = pda.Field(nullable=True)
    frac_credits_earned_year_2: pt.Series["Float32"] = pda.Field(nullable=True)
    frac_credits_earned_year_3: pt.Series["Float32"] = pda.Field(nullable=True)
    frac_credits_earned_year_4: pt.Series["Float32"] = pda.Field(nullable=True)
    year_of_enrollment_at_cohort_inst: pt.Series["int8"]
    term_is_while_student_enrolled_at_other_inst: pt.Series["bool"]
    frac_credits_earned: pt.Series["Float32"] = pda.Field(nullable=True)
    student_term_enrollment_intensity: pt.Series["string"]
    frac_courses_passed: pt.Series["float32"]
    frac_courses_completed: pt.Series["float32"]
    frac_courses_enrolled_at_other_institution_s_y: pt.Series["float32"]
    frac_courses_core_course_y: t.Optional[pt.Series["float32"]]
    frac_courses_core_course_n: t.Optional[pt.Series["float32"]]
    frac_courses_course_type_cc_cd: pt.Series["float32"]
    frac_courses_course_type_cu: pt.Series["float32"]
    frac_courses_course_type_cg: pt.Series["float32"]
    frac_courses_course_type_cc: pt.Series["float32"]
    frac_courses_course_type_cd: pt.Series["float32"]
    frac_courses_course_type_el: pt.Series["float32"]
    frac_courses_course_type_ab: pt.Series["float32"]
    frac_courses_course_type_ge: pt.Series["float32"]
    frac_courses_course_type_nc: pt.Series["float32"]
    frac_courses_course_type_o: pt.Series["float32"]
    frac_courses_delivery_method_f: pt.Series["float32"]
    frac_courses_delivery_method_o: pt.Series["float32"]
    frac_courses_delivery_method_h: pt.Series["float32"]
    frac_courses_math_or_english_gateway_m: pt.Series["float32"]
    frac_courses_math_or_english_gateway_e: pt.Series["float32"]
    frac_courses_math_or_english_gateway_na: pt.Series["float32"]
    frac_courses_co_requisite_course_y: pt.Series["float32"]
    frac_courses_co_requisite_course_n: pt.Series["float32"]
    frac_courses_course_instructor_employment_status_pt: pt.Series["float32"]
    frac_courses_course_instructor_employment_status_ft: pt.Series["float32"]
    frac_courses_course_instructor_rank_1: pt.Series["float32"]
    frac_courses_course_instructor_rank_2: pt.Series["float32"]
    frac_courses_course_instructor_rank_3: pt.Series["float32"]
    frac_courses_course_instructor_rank_4: pt.Series["float32"]
    frac_courses_course_instructor_rank_5: pt.Series["float32"]
    frac_courses_course_instructor_rank_6: pt.Series["float32"]
    frac_courses_course_instructor_rank_7: pt.Series["float32"]
    # NOTE: let's cover most likely values, not all possible values
    frac_courses_course_level_1: pt.Series["float32"]
    frac_courses_course_grade_a: pt.Series["float32"]
    frac_courses_grade_is_failing_or_withdrawal: pt.Series["float32"]
    frac_courses_grade_above_section_avg: pt.Series["float32"]
    frac_sections_students_passed: pt.Series["float32"]
    frac_sections_students_completed: pt.Series["float32"]
    student_pass_rate_above_sections_avg: pt.Series["bool"]
    student_completion_rate_above_sections_avg: pt.Series["bool"]
    # NOTE: let's not bother with the many cumulative features (for now?)
    # cumnum_terms_enrolled: pt.Series["int8"]
    # etc.
    num_courses_diff_prev_term: pt.Series["Int8"] = pda.Field(nullable=True)
    num_credits_earned_diff_prev_term: pt.Series["Float32"] = pda.Field(nullable=True)
    course_grade_numeric_mean_diff_prev_term: pt.Series["Float32"] = pda.Field(
        nullable=True
    )
    num_courses_diff_term_1_to_term_2: pt.Series["Int8"] = pda.Field(nullable=True)
    num_credits_earned_diff_term_1_to_term_2: pt.Series["Float32"] = pda.Field(
        nullable=True
    )
    course_grade_numeric_mean_diff_term_1_to_term_2: pt.Series["Float32"] = pda.Field(
        nullable=True
    )

    class Config:
        coerce = True
        unique_column_names = True
        add_missing_columns = False
        drop_invalid_rows = False
        unique = ["student_id", "term_id"]
