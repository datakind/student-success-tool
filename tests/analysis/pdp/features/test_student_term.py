import pandas as pd
import pytest

from student_success_tool.analysis.pdp.features import student_term

# @pytest.mark.parametrize(
#     ["df", "exp"],
#     [
#         (
#             pd.DataFrame(
#                 {
#                     "cohort": ["20-21", "20-21", "20-21", "22-23", "22-23"],
#                     "academic_year": ["20-21", "21-22", "23-24", "22-23", "23-24"],
#                     "num_credits_earned": [5.0, 10.0, 6.0, 0.0, 15.0],
#                     "num_credits_attempted": [10.0, 10.0, 8.0, 15.0, 15.0],
#                 }
#             ),
#             pd.DataFrame(
#                 {
#                     "year_of_enrollment_at_cohort_inst": [1, 2, 4, 1, 2],
#                     "frac_credits_earned": [0.5, 1.0, 0.75, 0.0, 1.0],
#                 }
#             ),
#         ),
#     ],
# )
# def test_add_student_term_features(df, exp):
#     obs = student_term.add_features(df)
#     assert isinstance(obs, pd.DataFrame) and not obs.empty
#     assert obs.equals(exp) or obs.compare(exp).empty


@pytest.mark.parametrize(
    ["df", "grp_cols", "agg_cols", "exp"],
    [
        (
            pd.DataFrame(
                {
                    "student_guid": ["123", "123", "123", "456", "456", "789"],
                    "term_id": [
                        "23-24 FALL",
                        "23-24 FALL",
                        "23-24 SPRING",
                        "23-24 FALL",
                        "23-24 FALL",
                        "23-24 SPRING",
                    ],
                    "course_type": ["CU", "CD", "CU", "CU", "CC", "CU"],
                    "course_level": [1, 0, 1, 2, 0, 1],
                }
            ),
            ["student_guid", "term_id"],
            ["course_type", "course_level"],
            pd.DataFrame(
                {
                    "student_guid": ["123", "123", "456", "789"],
                    "term_id": [
                        "23-24 FALL",
                        "23-24 SPRING",
                        "23-24 FALL",
                        "23-24 SPRING",
                    ],
                    "num_courses_course_type_CC": [0, 0, 1, 0],
                    "num_courses_course_type_CD": [1, 0, 0, 0],
                    "num_courses_course_type_CU": [1, 1, 1, 1],
                    "num_courses_course_level_0": [1, 0, 1, 0],
                    "num_courses_course_level_1": [1, 1, 0, 1],
                    "num_courses_course_level_2": [0, 0, 1, 0],
                }
            ),
        ),
    ],
)
def test_sum_dummy_cols_by_group(df, grp_cols, agg_cols, exp):
    obs = student_term.sum_dummy_cols_by_group(df, grp_cols=grp_cols, agg_cols=agg_cols)
    assert isinstance(obs, pd.DataFrame) and not obs.empty
    assert obs.equals(exp) or obs.compare(exp).empty


@pytest.mark.parametrize(
    ["df", "grp_cols", "agg_col_vals", "exp"],
    [
        (
            pd.DataFrame(
                {
                    "student_guid": ["123", "123", "123", "456", "456", "789"],
                    "term_id": [
                        "23-24 FALL",
                        "23-24 FALL",
                        "23-24 SPRING",
                        "23-24 FALL",
                        "23-24 FALL",
                        "23-24 SPRING",
                    ],
                    "course_type": ["CU", "CD", "CU", "CU", "CC", "CU"],
                    "course_level": [1, 0, 1, 2, 0, 1],
                    "grade": ["F", "F", "P", "W", "F", "P"],
                }
            ),
            ["student_guid", "term_id"],
            [
                ("course_type", ["CC", "CD"]),
                ("course_level", 0),
                ("grade", ["0", "1", "F", "W"]),
            ],
            pd.DataFrame(
                {
                    "student_guid": ["123", "123", "456", "789"],
                    "term_id": [
                        "23-24 FALL",
                        "23-24 SPRING",
                        "23-24 FALL",
                        "23-24 SPRING",
                    ],
                    "num_courses_course_type_CC|CD": [1, 0, 1, 0],
                    "num_courses_course_level_0": [1, 0, 1, 0],
                    "num_courses_grade_0|1|F|W": [2, 0, 2, 0],
                }
            ),
        ),
    ],
)
def test_sum_val_equal_cols_by_group(df, grp_cols, agg_col_vals, exp):
    obs = student_term.sum_val_equal_cols_by_group(
        df, grp_cols=grp_cols, agg_col_vals=agg_col_vals
    )
    assert isinstance(obs, pd.DataFrame) and not obs.empty
    assert obs.equals(exp) or obs.compare(exp).empty


@pytest.mark.parametrize(
    ["df", "grp_cols", "grade_col", "section_grade_col", "exp"],
    [
        (
            pd.DataFrame(
                {
                    "sid": ["123", "123", "123", "456", "456"],
                    "tid": ["22-23 FA", "22-23 FA", "22-23 SP", "22-23 SP", "22-23 SP"],
                    "course_grade": [4, 3, 1, pd.NA, 4],
                    "section_course_grade_mean": [3.25, 3.0, 2.5, 3.0, 3.5],
                }
            ).astype({"course_grade": "Int8"}),
            ["sid", "tid"],
            "course_grade",
            "section_course_grade_mean",
            pd.DataFrame(
                {
                    "sid": ["123", "123", "456"],
                    "tid": ["22-23 FA", "22-23 SP", "22-23 SP"],
                    "num_courses_grade_above_section_avg": [1, 0, 1],
                }
            ),
        ),
    ],
)
def test_multicol_aggs_by_group(df, grp_cols, grade_col, section_grade_col, exp):
    obs = student_term.multicol_aggs_by_group(
        df,
        grp_cols=grp_cols,
        grade_col=grade_col,
        section_grade_col=section_grade_col,
    )
    assert isinstance(obs, pd.DataFrame) and not obs.empty
    assert obs.equals(exp) or obs.compare(exp).empty


@pytest.mark.parametrize(
    ["df", "ccol", "acol", "exp"],
    [
        (
            pd.DataFrame(
                {
                    "cohort": ["2021-22", "2021-22", "2022-23"],
                    "academic_year": ["2021-22", "2022-23", "2024-25"],
                }
            ),
            "cohort",
            "academic_year",
            pd.Series([1, 2, 3], dtype="Int16"),
        ),
    ],
)
def test_year_of_enrollment_at_cohort_inst(df, ccol, acol, exp):
    obs = student_term.year_of_enrollment_at_cohort_inst(
        df, cohort_col=ccol, academic_col=acol
    )
    assert isinstance(obs, pd.Series) and not obs.empty
    assert obs.equals(exp) or obs.compare(exp).empty


@pytest.mark.parametrize(
    ["df", "numer_col", "denom_col", "exp"],
    [
        (
            pd.DataFrame(
                {
                    "num_courses": [1, 2, 4, 5, 7],
                    "num_courses_passed": [1, 1, 3, 4, 0],
                }
            ),
            "num_courses_passed",
            "num_courses",
            pd.Series([1.0, 0.5, 0.75, 0.8, 0.0]),
        ),
    ],
)
def test_compute_frac_courses(df, numer_col, denom_col, exp):
    obs = student_term.compute_frac_courses(
        df, numer_col=numer_col, denom_col=denom_col
    )
    assert isinstance(obs, pd.Series) and not obs.empty
    assert obs.equals(exp) or obs.compare(exp).empty


@pytest.mark.parametrize(
    ["df", "student_col", "sections_col", "exp"],
    [
        (
            pd.DataFrame(
                {
                    "frac_courses_passed": [1.0, 0.5, 0.75],
                    "frac_sections_students_passed": [0.9, 0.5, 0.8],
                }
            ),
            "frac_courses_passed",
            "frac_sections_students_passed",
            pd.Series([True, False, False]),
        ),
    ],
)
def test_student_rate_above_sections_avg(df, student_col, sections_col, exp):
    obs = student_term.student_rate_above_sections_avg(
        df, student_col=student_col, sections_col=sections_col
    )
    assert isinstance(obs, pd.Series) and not obs.empty
    assert obs.equals(exp) or obs.compare(exp).empty
