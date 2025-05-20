import pandas as pd
import pytest

from student_success_tool.preprocessing.features.pdp import section


@pytest.mark.parametrize(
    ["df", "section_id_cols", "exp"],
    [
        (
            pd.DataFrame(
                {
                    "year": ["23-24", "23-24", "23-24", "23-24", "23-24"],
                    "term": ["FA", "FA", "FA", "FA", "FA"],
                    "course_id": ["X01", "X01", "Y101", "Y101", "Y101"],
                    "section_id": ["M01", "M01", "B01", "B01", "B02"],
                    "student_id": ["123", "456", "123", "456", "789"],
                    "course_grade_numeric": [4, 3, 2, pd.NA, 4],
                    "course_passed": [True, False, True, pd.NA, False],
                    "course_completed": [True, True, True, False, True],
                }
            ).astype({"course_grade_numeric": "Int8", "course_passed": "boolean"}),
            ["year", "term", "course_id", "section_id"],
            pd.DataFrame(
                {
                    "year": ["23-24", "23-24", "23-24", "23-24", "23-24"],
                    "term": ["FA", "FA", "FA", "FA", "FA"],
                    "course_id": ["X01", "X01", "Y101", "Y101", "Y101"],
                    "section_id": ["M01", "M01", "B01", "B01", "B02"],
                    "student_id": ["123", "456", "123", "456", "789"],
                    "course_grade_numeric": [4, 3, 2, pd.NA, 4],
                    "course_passed": [True, False, True, pd.NA, False],
                    "course_completed": [True, True, True, False, True],
                    "section_num_students_enrolled": [2, 2, 2, 2, 1],
                    "section_num_students_passed": [1, 1, 1, 1, 0],
                    "section_num_students_completed": [2, 2, 1, 1, 1],
                    "section_course_grade_numeric_mean": [3.5, 3.5, 2.0, 2.0, 4.0],
                }
            ).astype(
                {
                    "course_grade_numeric": "Int8",
                    "course_passed": "boolean",
                    "section_course_grade_numeric_mean": "Float32",
                }
            ),
        ),
    ],
)
def test_add_section_features(df, section_id_cols, exp):
    obs = section.add_features(df, section_id_cols=section_id_cols)
    assert isinstance(obs, pd.DataFrame) and not obs.empty
    assert obs.equals(exp) or obs.compare(exp).empty
