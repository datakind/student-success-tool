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
                    "Cohort": ["2024-25", "2023-24"],
                    "Cohort Term": ["FALL", "SPRING"],
                    "Time to Credential": [2.5, 3.0],
                    "Attendance Status Term 1": [
                        "FIRST-TIME FULL-TIME",
                        "TRANSFER-IN PART-TIME",
                    ],
                    "Number of Credits Attempted Year 1": [10.0, 5.0],
                    "Number of Credits Earned Year 1": [10.0, 5.0],
                    "Gateway Math Status": ["R", "N"],
                    "GPA Group Year 1": [3.0, 4.0],
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
                    "Cohort": ["2024-25", "2023-24"],
                    "Cohort Term": ["FALL", "SPRING"],
                    "GPA Group Year 1": [3.0, 4.0],
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
                    "Academic Year": ["2020-21", "2020-21", "2021-22"],
                    "Academic Term": ["FALL", "SPRING", "FALL"],
                    "Course Prefix": ["MATH", "MATH", "PHYS"],
                    "Course Number": ["101", "202", "303"],
                    "Course Name": ["NAME1", "NAME2", "NAME3"],
                    "Course Type": ["CU", "CU", "CC"],
                    "Grade": ["4", "1", "W"],
                    "Core Course": ["Y", "N", "Y"],
                    "Core Course Type": ["TYPE1", "TYPE2", "TYPE3"],
                    "Core Competency Completed": ["Y", None, None],
                    "Cohort": ["2020-21", "2020-21", "2020-21"],
                    "Cohort Term": ["FALL", "FALL", "SPRING"],
                    "Student Age": ["20 AND YOUNGER", ">20 - 24", "OLDER THAN 24"],
                    "Race": ["WHITE", "HISPANIC", "ASIAN"],
                    "Ethnicity": ["N", "H", "N"],
                    "Gender": ["M", "F", "X"],
                    "Credential Engine Identifier": [None, None, None],
                    "Enrollment Record at Other Institution(s) STATE(s)": [
                        "VT",
                        None,
                        None,
                    ],
                    "Enrollment Record at Other Institution(s) CARNEGIE(s)": [
                        "Doctoral Universities",
                        None,
                        None,
                    ],
                    "Enrollment Record at Other Institution(s) LOCALE(s)": [
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
