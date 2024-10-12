import pandas as pd
import pytest

from student_success_tool.analysis.pdp import dataops


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
    obs = dataops.infer_first_term_of_year(series)
    assert obs == exp


@pytest.mark.parametrize(
    ["series", "exp"],
    [
        (pd.Series(["WI", "SP", "FA", "SU"], dtype="category"), 4),
        (pd.Series(["SU", "FA", "WI"], dtype="category"), 3),
    ],
)
def test_infer_num_terms_in_year(series, exp):
    obs = dataops.infer_num_terms_in_year(series)
    assert obs == exp


@pytest.mark.parametrize(
    ["df", "num_terms_checkin", "exp"],
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
                }
            ),
            None,
            pd.DataFrame(
                {
                    "cohort": ["2024-25", "2023-24"],
                    "cohort_term": ["FALL", "SPRING"],
                    "gpa_group_year_1": [3.0, 4.0],
                }
            ),
        ),
        (
            pd.DataFrame(
                {
                    "cohort": ["2024-25", "2023-24"],
                    "cohort_term": ["FALL", "SPRING"],
                    "gpa_group_year_1": [3.0, 4.0],
                }
            ),
            1,
            pd.DataFrame(
                {
                    "cohort": ["2024-25", "2023-24"],
                    "cohort_term": ["FALL", "SPRING"],
                }
            ),
        ),
    ],
)
def test_standardize_cohort_dataset(df, num_terms_checkin, exp):
    obs = dataops.standardize_cohort_dataset(df, num_terms_checkin=num_terms_checkin)
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
                }
            ),
        ),
    ],
)
def test_standardize_course_dataset(df, exp):
    obs = dataops.standardize_course_dataset(df)
    assert isinstance(obs, pd.DataFrame) and not obs.empty
    assert obs.equals(exp) or obs.compare(exp).empty
