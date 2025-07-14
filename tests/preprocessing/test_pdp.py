import os

import numpy as np
import pandas as pd
import pytest

from student_success_tool import dataio, preprocessing

SYNTHETIC_DATA_PATH = "synthetic-data/pdp"


@pytest.mark.parametrize(
    ["series", "exp"],
    [
        (
            pd.Series(
                ["WI", "SP", "FA", "SU"],
                dtype=pd.CategoricalDtype(["FA", "WI", "SP", "SU"], ordered=True),
            ),
            "FA",
        ),
        (
            pd.Series(
                ["FA", "SU", "SP"],
                dtype=pd.CategoricalDtype(["SU", "FA", "SP"], ordered=True),
            ),
            "SU",
        ),
    ],
)
def test_infer_first_term_of_year(series, exp):
    obs = preprocessing.pdp.infer_first_term_of_year(series)
    assert obs == exp


@pytest.mark.parametrize(
    ["series", "exp"],
    [
        (pd.Series(["WI", "SP", "FA", "SU"], dtype="category"), 4),
        (pd.Series(["SU", "FA", "WI"], dtype="category"), 3),
    ],
)
def test_infer_num_terms_in_year(series, exp):
    obs = preprocessing.pdp.infer_num_terms_in_year(series)
    assert obs == exp


@pytest.mark.parametrize(
    ["df", "exp"],
    [
        (
            pd.DataFrame(
                {
                    "cohort": ["2024-25", "2023-24"],
                    "cohort_term": ["FALL", "SPRING"],
                    "time_to_credential": [2.5, 3.0],
                    "attendance_status_term_1": [
                        "FIRST-TIME FULL-TIME",
                        "TRANSFER-IN PART-TIME",
                    ],
                    "number_of_credits_attempted_year_1": [10.0, 5.0],
                    "number_of_credits_earned_year_1": [10.0, 5.0],
                    "gateway_math_status": ["R", "N"],
                    "gpa_group_year_1": [3.0, 4.0],
                    "most_recent_bachelors_at_other_institution_state": ["VT", "IL"],
                }
            ),
            pd.DataFrame(
                {
                    "cohort": ["2024-25", "2023-24"],
                    "cohort_term": ["FALL", "SPRING"],
                    "number_of_credits_attempted_year_1": [10.0, 5.0],
                    "number_of_credits_earned_year_1": [10.0, 5.0],
                    "gpa_group_year_1": [3.0, 4.0],
                    "years_to_latest_associates_at_cohort_inst": [pd.NA, pd.NA],
                    "years_to_latest_certificate_at_cohort_inst": [pd.NA, pd.NA],
                    "years_to_latest_associates_at_other_inst": [pd.NA, pd.NA],
                    "years_to_latest_certificate_at_other_inst": [pd.NA, pd.NA],
                    "first_year_to_associates_at_cohort_inst": [pd.NA, pd.NA],
                    "first_year_to_certificate_at_cohort_inst": [pd.NA, pd.NA],
                    "first_year_to_associates_at_other_inst": [pd.NA, pd.NA],
                    "first_year_to_certificate_at_other_inst": [pd.NA, pd.NA],
                }
            ).astype(
                {
                    col: "Int8"
                    for col in [
                        "years_to_latest_associates_at_cohort_inst",
                        "years_to_latest_certificate_at_cohort_inst",
                        "years_to_latest_associates_at_other_inst",
                        "years_to_latest_certificate_at_other_inst",
                        "first_year_to_associates_at_cohort_inst",
                        "first_year_to_certificate_at_cohort_inst",
                        "first_year_to_associates_at_other_inst",
                        "first_year_to_certificate_at_other_inst",
                    ]
                }
            ),
        ),
    ],
)
def test_standardize_cohort_dataset(df, exp):
    obs = preprocessing.pdp.standardize_cohort_dataset(df)
    assert isinstance(obs, pd.DataFrame) and not obs.empty
    assert obs.equals(exp) or obs.compare(exp).empty


@pytest.mark.parametrize(
    ["df", "exp"],
    [
        (
            pd.DataFrame(
                {
                    "academic_year": ["2020-21", "2020-21", "2021-22"],
                    "academic_term": ["FALL", "SPRING", "FALL"],
                    "course_prefix": ["MATH", "MATH", "PHYS"],
                    "course_number": ["101", "202", "303"],
                    "course_name": ["NAME1", "NAME2", "NAME3"],
                    "course_type": ["CU", "CU", "CC"],
                    "grade": ["4", "1", "W"],
                    "core_course": ["Y", "N", "Y"],
                    "core_course_type": ["TYPE1", "TYPE2", "TYPE3"],
                    "core_competency_completed": ["Y", None, None],
                    "cohort": ["2020-21", "2020-21", "2020-21"],
                    "cohort_term": ["FALL", "FALL", "SPRING"],
                    "student_age": ["20 AND YOUNGER", ">20 - 24", "OLDER THAN 24"],
                    "race": ["WHITE", "HISPANIC", "ASIAN"],
                    "ethnicity": ["N", "H", "N"],
                    "gender": ["M", "F", "X"],
                    "credential_engine_identifier": [None, None, None],
                    "enrollment_record_at_other_institution_s_state_s": [
                        "VT",
                        None,
                        None,
                    ],
                    "enrollment_record_at_other_institution_s_carnegie_s": [
                        "Doctoral Universities",
                        None,
                        None,
                    ],
                    "enrollment_record_at_other_institution_s_locale_s": [
                        "Urban",
                        None,
                        None,
                    ],
                }
            ),
            pd.DataFrame(
                {
                    "academic_year": ["2020-21", "2020-21", "2021-22"],
                    "academic_term": ["FALL", "SPRING", "FALL"],
                    "course_prefix": ["MATH", "MATH", "PHYS"],
                    "course_number": ["101", "202", "303"],
                    "course_type": ["CU", "CU", "CC"],
                    "grade": ["4", "1", "W"],
                    "core_course": ["Y", "N", "Y"],
                    "term_program_of_study": [pd.NA, pd.NA, pd.NA],
                }
            ),
        ),
    ],
)
def test_standardize_course_dataset(df, exp):
    obs = preprocessing.pdp.standardize_course_dataset(df)
    assert isinstance(obs, pd.DataFrame) and not obs.empty
    assert obs.equals(exp) or obs.compare(exp).empty


@pytest.mark.parametrize(
    ["df", "col_val_dtypes", "exp"],
    [
        (
            pd.DataFrame(),
            {"a": (None, "Int8")},
            pd.DataFrame(columns=["a"], dtype="Int8"),
        ),
        (
            pd.DataFrame({"a": [1, 2, 3]}),
            {"b": (None, "string")},
            pd.DataFrame({"a": [1, 2, 3], "b": [pd.NA, pd.NA, pd.NA]}).astype(
                {"b": "string"}
            ),
        ),
        (
            pd.DataFrame({"a": [1, 2, 3]}),
            {"a": (None, "Int8"), "b": ("NA", "string")},
            pd.DataFrame({"a": [1, 2, 3], "b": ["NA", "NA", "NA"]}).astype(
                {"b": "string"}
            ),
        ),
    ],
)
def test_add_empty_cols_if_missing(df, col_val_dtypes, exp):
    obs = preprocessing.pdp.add_empty_cols_if_missing(df, col_val_dtypes=col_val_dtypes)
    assert pd.testing.assert_frame_equal(obs, exp) is None


@pytest.mark.parametrize(
    ["df", "exp"],
    [
        (
            pd.DataFrame(
                {
                    "year_of_enrollment_at_cohort_inst": [1, 2, 3, 4],
                    "cumnum_terms_enrolled": [2, 3, 4, 5],
                    "term_id": [
                        "2020-21 FALL",
                        "2020-21 WINTER",
                        "2020-21 SPRING",
                        "2020-21 SUMMER",
                    ],
                    "course_ids": [
                        ["X101", "Y101"],
                        ["X102", "Y101", "Z201"],
                        ["X101", "Y102"],
                        ["X101", "Y101", "X102", "Y202", "Z101"],
                    ],
                    "retention": [False, True, True, True],
                    "years_to_bachelors_at_cohort_inst": [pd.NA, pd.NA, 4, 3],
                    "years_to_bachelor_at_other_inst": [pd.NA, 3, 2, 2],
                    "first_year_to_bachelors_at_cohort_inst": [pd.NA, pd.NA, 4, 4],
                    "first_year_to_bachelor_at_other_inst": [pd.NA, pd.NA, pd.NA, 6],
                    "years_to_latest_associates_at_cohort_inst": [pd.NA, 3, 2, 2],
                    "years_to_latest_associates_at_other_inst": [pd.NA, pd.NA, 2, 1],
                    "first_year_to_associates_at_cohort_inst": [pd.NA, pd.NA, 4, 3],
                    "first_year_to_associates_at_other_inst": [pd.NA, pd.NA, 3, 2],
                    "years_to_associates_or_certificate_at_cohort_inst": [
                        pd.NA,
                        pd.NA,
                        pd.NA,
                        5,
                    ],
                    "years_to_associates_or_certificate_at_other_inst": [
                        pd.NA,
                        3,
                        2,
                        4,
                    ],
                    "first_year_to_associates_or_certificate_at_cohort_inst": [
                        pd.NA,
                        2,
                        2,
                        5,
                    ],
                    "first_year_to_associates_or_certificate_at_other_inst": [
                        4,
                        1,
                        5,
                        3,
                    ],
                    "first_year_to_certificate_at_cohort_inst": [pd.NA, 2, 2, 5],
                    "first_year_to_certificate_at_other_inst": [pd.NA, 2, 2, 2],
                    "years_to_latest_certificate_at_cohort_inst": [pd.NA, 2, 2, 5],
                    "years_to_latest_certificate_at_other_inst": [pd.NA, 3, 3, 3],
                    "years_of_last_enrollment_at_cohort_institution": [
                        pd.NA,
                        pd.NA,
                        pd.NA,
                        3,
                    ],
                    "years_of_last_enrollment_at_other_institution": [1, 2, 3, 4],
                    "frac_credits_earned_year_1": [1.0, 0.5, 0.75, 0.9],
                    "frac_credits_earned_year_2": [0.9, 0.75, 0.8, 0.85],
                    "num_courses_diff_term_2_to_term_3": [0.0, 1.0, -1.0, 0.0],
                    "num_courses_diff_term_3_to_term_4": [1.0, -1.0, 0.0, 1.0],
                    "cumsum_num_credits_earned": [10, 15, 20, 9],
                    "cummax_in_12_creds_took_subject_area_51": [
                        False,
                        True,
                        True,
                        True,
                    ],
                }
            ).astype(
                {
                    "years_to_bachelors_at_cohort_inst": "Int8",
                    "years_to_bachelor_at_other_inst": "Int8",
                    "first_year_to_bachelors_at_cohort_inst": "Int8",
                    "first_year_to_bachelor_at_other_inst": "Int8",
                    "years_to_latest_associates_at_cohort_inst": "Int8",
                    "years_to_latest_associates_at_other_inst": "Int8",
                    "first_year_to_associates_at_cohort_inst": "Int8",
                    "first_year_to_associates_at_other_inst": "Int8",
                    "years_to_associates_or_certificate_at_cohort_inst": "Int8",
                    "years_to_associates_or_certificate_at_other_inst": "Int8",
                    "first_year_to_associates_or_certificate_at_cohort_inst": "Int8",
                    "first_year_to_associates_or_certificate_at_other_inst": "Int8",
                    "cummax_in_12_creds_took_subject_area_51": "boolean",
                    "years_of_last_enrollment_at_cohort_institution": "Int8",
                    "years_of_last_enrollment_at_other_institution": "Int8",
                    "years_to_latest_certificate_at_cohort_inst": "Int8",
                    "years_to_latest_certificate_at_other_inst": "Int8",
                    "first_year_to_certificate_at_cohort_inst": "Int8",
                    "first_year_to_certificate_at_other_inst": "Int8",
                }
            ),
            pd.DataFrame(
                {
                    "year_of_enrollment_at_cohort_inst": [1, 2, 3, 4],
                    "cumnum_terms_enrolled": [2, 3, 4, 5],
                    "first_year_to_certificate_at_cohort_inst": [
                        pd.NA,
                        pd.NA,
                        2,
                        pd.NA,
                    ],
                    "first_year_to_certificate_at_other_inst": [pd.NA, pd.NA, 2, 2],
                    "years_to_latest_certificate_at_cohort_inst": [
                        pd.NA,
                        pd.NA,
                        2,
                        pd.NA,
                    ],
                    "years_to_latest_certificate_at_other_inst": [
                        pd.NA,
                        pd.NA,
                        pd.NA,
                        3,
                    ],
                    "frac_credits_earned_year_1": [np.nan, 0.5, 0.75, 0.9],
                    "frac_credits_earned_year_2": [np.nan, np.nan, 0.8, 0.85],
                    "num_courses_diff_term_2_to_term_3": [np.nan, 1.0, -1.0, 0.0],
                    "num_courses_diff_term_3_to_term_4": [np.nan, np.nan, 0.0, 1.0],
                    "cumsum_num_credits_earned": [10, 15, 20, 9],
                    "cummax_in_12_creds_took_subject_area_51": [
                        np.nan,
                        True,
                        True,
                        np.nan,
                    ],
                }
            ).astype(
                {
                    "first_year_to_certificate_at_cohort_inst": "Int8",
                    "first_year_to_certificate_at_other_inst": "Int8",
                    "years_to_latest_certificate_at_cohort_inst": "Int8",
                    "years_to_latest_certificate_at_other_inst": "Int8",
                }
            ),
        ),
    ],
)
def test_clean_up_labeled_dataset_cols_and_vals(df, exp):
    obs = preprocessing.pdp.clean_up_labeled_dataset_cols_and_vals(df)
    assert isinstance(obs, pd.DataFrame) and not obs.empty
    print("observed cols", obs.columns)
    print("expected cols", exp.columns)
    assert obs.equals(exp) or obs.compare(exp).empty


@pytest.mark.parametrize(
    ["cohort_file_name", "course_file_name"],
    [
        (
            "INSTXYZ_STUDENT_SEMESTER_AR_DEIDENTIFIED.csv",
            "INSTXYZ_COURSE_LEVEL_AR_DEID.csv",
        ),
    ],
)
def test_make_student_term_dataset_against_checked_in_sample(
    cohort_file_name, course_file_name
):
    full_cohort_file_path = os.path.join(SYNTHETIC_DATA_PATH, cohort_file_name)
    cohort = dataio.pdp.read_raw_cohort_data(
        file_path=full_cohort_file_path,
        schema=dataio.schemas.pdp.RawPDPCohortDataSchema,
    )
    assert isinstance(cohort, pd.DataFrame)
    assert not cohort.empty

    full_course_file_path = os.path.join(SYNTHETIC_DATA_PATH, course_file_name)
    course = dataio.pdp.read_raw_course_data(
        file_path=full_course_file_path,
        schema=dataio.schemas.pdp.RawPDPCourseDataSchema,
        dttm_format="%Y-%m-%d",
    )
    assert isinstance(course, pd.DataFrame)
    assert not course.empty
    df = preprocessing.pdp.make_student_term_dataset(cohort, course)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
