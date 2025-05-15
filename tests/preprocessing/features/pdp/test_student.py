import pandas as pd
import pytest

from student_success_tool.preprocessing.features.pdp import student


@pytest.mark.parametrize(
    ["df", "first_term_of_year", "exp"],
    [
        (
            pd.DataFrame(
                {
                    "student_guid": ["123", "456"],
                    "cohort": ["2020-21", "2021-22"],
                    "cohort_term": ["FALL", "SUMMER"],
                    "program_of_study_term_1": ["24.0101", "27.01"],
                    "program_of_study_year_1": ["24.0101", "27.05"],
                    "gpa_group_term_1": [4.00, 3.00],
                    "gpa_group_year_1": [3.5, 3.5],
                    "number_of_credits_attempted_year_1": [15.0, 12.0],
                    "number_of_credits_earned_year_1": [12.0, 12.0],
                    "pell_status_first_year": ["Y", "N"],
                }
            ),
            "FALL",
            pd.DataFrame(
                {
                    "student_guid": ["123", "456"],
                    "cohort": ["2020-21", "2021-22"],
                    "cohort_term": ["FALL", "SUMMER"],
                    "program_of_study_term_1": ["24.0101", "27.01"],
                    "program_of_study_year_1": ["24.0101", "27.05"],
                    "gpa_group_term_1": [4.00, 3.00],
                    "gpa_group_year_1": [3.5, 3.5],
                    "number_of_credits_attempted_year_1": [15.0, 12.0],
                    "number_of_credits_earned_year_1": [12.0, 12.0],
                    "pell_status_first_year": ["Y", "N"],
                    "cohort_id": ["2020-21 FALL", "2021-22 SUMMER"],
                    "cohort_start_dt": ["2020-09-01", "2022-06-01"],
                    "student_program_of_study_area_term_1": ["24", "27"],
                    "student_program_of_study_area_year_1": ["24", "27"],
                    "student_program_of_study_changed_term_1_to_year_1": [False, True],
                    "student_program_of_study_area_changed_term_1_to_year_1": [
                        False,
                        False,
                    ],
                    "student_is_pell_recipient_first_year": [True, False],
                    "diff_gpa_term_1_to_year_1": [-0.5, 0.5],
                    "frac_credits_earned_year_1": [0.8, 1.0],
                }
            ),
        ),
    ],
)
def test_add_student_features(df, first_term_of_year, exp):
    obs = student.add_features(df, first_term_of_year=first_term_of_year)
    assert isinstance(obs, pd.DataFrame) and not obs.empty
    assert obs.equals(exp) or obs.compare(exp).empty


@pytest.mark.parametrize(
    ["df", "col", "exp"],
    [
        (
            pd.DataFrame({"program_cip": ["240012.0", "519999", "9.1000", "X"]}),
            "program_cip",
            pd.Series(["24", "51", "9", pd.NA], dtype="string"),
        ),
    ],
)
def test_student_program_of_study_area(df, col, exp):
    obs = student.student_program_of_study_area(df, col=col)
    assert isinstance(obs, pd.Series) and not obs.empty
    assert obs.equals(exp) or obs.compare(exp).empty


@pytest.mark.parametrize(
    ["df", "term_col", "year_col", "exp"],
    [
        (
            pd.DataFrame(
                {
                    "posa_term_1": [24, 51, 9, pd.NA],
                    "posa_year_1": [24, 42, 9, pd.NA],
                },
                dtype="Int8",
            ),
            "posa_term_1",
            "posa_year_1",
            pd.Series([False, True, False, pd.NA], dtype="boolean"),
        ),
    ],
)
def test_student_program_of_study_changed_term_1_to_year_1(df, term_col, year_col, exp):
    obs = student.student_program_of_study_changed_term_1_to_year_1(
        df, term_col=term_col, year_col=year_col
    )
    assert isinstance(obs, pd.Series) and not obs.empty
    assert obs.equals(exp) or obs.compare(exp).empty


@pytest.mark.parametrize(
    ["df", "exp"],
    [
        (
            pd.DataFrame({"pell_status_first_year": ["Y", "N", pd.NA]}),
            pd.Series([True, False, pd.NA], dtype="boolean"),
        )
    ],
)
def student_is_pell_recipient_first_year(df, exp):
    obs = student.student_is_pell_recipient_first_year(
        df, pell_col="pell_status_first_year"
    )
    assert isinstance(obs, pd.Series) and not obs.empty
    assert obs.equals(exp) or obs.compare(exp).empty


@pytest.mark.parametrize(
    ["df", "exp"],
    [
        (
            pd.DataFrame(
                {"gpa_year_1": [1.0, 2.0, 3.5], "gpa_term_1": [2.0, 1.5, 4.0]}
            ),
            pd.Series([-1.0, 0.5, -0.5], dtype="float"),
        )
    ],
)
def test_diff_gpa_term_1_to_year_1(df, exp):
    obs = student.diff_gpa_term_1_to_year_1(
        df, term_col="gpa_term_1", year_col="gpa_year_1"
    )
    assert isinstance(obs, pd.Series) and not obs.empty
    assert obs.equals(exp) or obs.compare(exp).empty
