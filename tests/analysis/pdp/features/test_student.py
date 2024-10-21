import pandas as pd
import pytest

from student_success_tool.analysis.pdp.features import student


@pytest.mark.parametrize(
    ["df", "institution_state", "exp"],
    [
        (
            pd.DataFrame(
                {
                    "student_guid": ["123", "456"],
                    "program_of_study_term_1": ["24.0101", "27.01"],
                    "program_of_study_year_1": ["24.0101", "27.05"],
                    "gpa_group_term_1": [4.00, 3.00],
                    "gpa_group_year_1": [3.5, 3.5],
                    "most_recent_last_enrollment_at_other_institution_state": [
                        "VT",
                        "PA",
                    ],
                }
            ),
            "VT",
            pd.DataFrame(
                {
                    "student_guid": ["123", "456"],
                    "program_of_study_term_1": ["24.0101", "27.01"],
                    "program_of_study_year_1": ["24.0101", "27.05"],
                    "gpa_group_term_1": [4.00, 3.00],
                    "gpa_group_year_1": [3.5, 3.5],
                    "most_recent_last_enrollment_at_other_institution_state": [
                        "VT",
                        "PA",
                    ],
                    # "student_has_prior_enrollment_at_other_inst": [True, True],
                    # "student_prior_enrollment_at_other_inst_was_in_state": [
                    #     True,
                    #     False,
                    # ],
                    "student_program_of_study_area_term_1": [24, 27],
                    "student_program_of_study_area_year_1": [24, 27],
                    "student_program_of_study_changed_first_year": [False, True],
                    "student_program_of_study_area_changed_first_year": [False, False],
                    "diff_gpa_year_1_to_term_1": [-0.5, 0.5]
                }
            ),
        ),
        (
            pd.DataFrame(
                {
                    "student_guid": ["123", "456"],
                    "program_of_study_term_1": ["24.0101", "27.01"],
                    "most_recent_last_enrollment_at_other_institution_state": [
                        "VT",
                        pd.NA,
                    ],
                }
            ),
            "VT",
            pd.DataFrame(
                {
                    "student_guid": ["123", "456"],
                    "program_of_study_term_1": ["24.0101", "27.01"],
                    "most_recent_last_enrollment_at_other_institution_state": [
                        "VT",
                        pd.NA,
                    ],
                    # "student_has_prior_enrollment_at_other_inst": [True, False],
                    # "student_prior_enrollment_at_other_inst_was_in_state": [
                    #     True,
                    #     pd.NA,
                    # ],
                    "student_program_of_study_area_term_1": [24, 27],
                }
            ),
        ),
    ],
)
def test_add_student_features(df, institution_state, exp):
    obs = student.add_features(df, institution_state=institution_state)
    assert isinstance(obs, pd.DataFrame) and not obs.empty
    assert obs.equals(exp) or obs.compare(exp).empty


@pytest.mark.parametrize(
    ["df", "col", "exp"],
    [
        (
            pd.DataFrame({"program_cip": ["240012.0", "519999", "9.1000", "X"]}),
            "program_cip",
            pd.Series([24, 51, 9, pd.NA], dtype="Int8"),
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
def test_student_program_of_study_changed_first_year(df, term_col, year_col, exp):
    obs = student.student_program_of_study_changed_first_year(
        df, term_col=term_col, year_col=year_col
    )
    assert isinstance(obs, pd.Series) and not obs.empty
    assert obs.equals(exp) or obs.compare(exp).empty


@pytest.mark.parametrize(
    ["df", "exp"],
    [
        (pd.DataFrame({'gpa_year_1': [1.0, 2.0, 3.5],
                       'gpa_term_1': [2.0, 1.5, 4.0]}),
        pd.Series([-1.0, 0.5, -0.5], dtype='float')
        )
    ]
)
def test_diff_gpa_year_1_to_term_1(df, exp):
    obs = student.diff_gpa_year_1_to_term_1(df, term_col = "gpa_term_1", year_col = "gpa_year_1")
    assert isinstance(obs, pd.Series) and not obs.empty
    assert obs.equals(exp) or obs.compare(exp).empty