import pandas as pd
import pytest

from student_success_tool.preprocessing.features.pdp import student_term

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
    ["df", "grp_cols", "exp"],
    [
        (
            pd.DataFrame(
                {
                    "student_guid": ["123", "123", "456", "789"],
                    "term_id": [
                        "23-24 FALL",
                        "23-24 SPRING",
                        "23-24 FALL",
                        "23-24 SPRING",
                    ],
                    "num_courses_course_type_CC|CD": [0, 3, 1, 0],
                    "num_courses_course_id_eng_101": [0, 0, 0, 1],
                    "num_courses_course_subject_area_51": [2, 1, 1, 1],
                }
            ),
            ["student_guid", "term_id"],
            pd.DataFrame(
                {
                    "student_guid": ["123", "123", "456", "789"],
                    "term_id": [
                        "23-24 FALL",
                        "23-24 SPRING",
                        "23-24 FALL",
                        "23-24 SPRING",
                    ],
                    "took_course_id_eng_101": [False, False, False, True],
                    "took_course_subject_area_51": [True, True, True, True],
                }
            ),
        ),
    ],
)
def test_equal_cols_by_group(df, grp_cols, exp):
    obs = student_term.equal_cols_by_group(df, grp_cols=grp_cols)
    print("obs columns", obs.columns)
    print("exp cols", exp.columns)
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
                ("course_level", [2, 3]),
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
                    "num_courses_course_level_2|3": [0, 0, 1, 0],
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
    [
        "df",
        "min_passing_grade",
        "grp_cols",
        "grade_col",
        "grade_numeric_col",
        "section_grade_numeric_col",
        "exp",
    ],
    [
        (
            pd.DataFrame(
                {
                    "sid": ["123", "123", "123", "123", "456", "456"],
                    "tid": [
                        "22-23 FA",
                        "22-23 FA",
                        "22-23 FA",
                        "22-23 SP",
                        "22-23 SP",
                        "22-23 SP",
                    ],
                    "grade": ["4", "3", "F", "1", pd.NA, "4"],
                    "grade_num": [4.0, 3.0, pd.NA, 1.0, pd.NA, 4.0],
                    "section_grade_num_mean": [3.25, 3.0, 2.75, 2.5, 3.0, 3.5],
                }
            ).astype({"grade": "string", "grade_num": "Float32"}),
            1.0,
            ["sid", "tid"],
            "grade",
            "grade_num",
            "section_grade_num_mean",
            pd.DataFrame(
                {
                    "sid": ["123", "123", "456"],
                    "tid": ["22-23 FA", "22-23 SP", "22-23 SP"],
                    "num_courses_grade_is_failing_or_withdrawal": [1, 0, 0],
                    "num_courses_grade_above_section_avg": [1, 0, 1],
                }
            ),
        ),
    ],
)
def test_multicol_grade_aggs_by_group(
    df,
    min_passing_grade,
    grp_cols,
    grade_col,
    grade_numeric_col,
    section_grade_numeric_col,
    exp,
):
    obs = student_term.multicol_grade_aggs_by_group(
        df,
        min_passing_grade=min_passing_grade,
        grp_cols=grp_cols,
        grade_col=grade_col,
        grade_numeric_col=grade_numeric_col,
        section_grade_numeric_col=section_grade_numeric_col,
    )
    assert isinstance(obs, pd.DataFrame) and not obs.empty
    assert obs.equals(exp) or obs.compare(exp).empty


@pytest.mark.parametrize(
    ["df", "ccol", "tcol", "exp"],
    [
        (
            pd.DataFrame(
                {
                    "cohort_start_dt": ["2019-09-01", "2019-09-01", "2021-02-01"],
                    "term_start_dt": ["2020-02-01", "2020-09-01", "2023-09-01"],
                },
                dtype="datetime64[s]",
            ),
            "cohort_start_dt",
            "term_start_dt",
            pd.Series([1, 2, 3], dtype="Int8"),
        ),
    ],
)
def test_year_of_enrollment_at_cohort_inst(df, ccol, tcol, exp):
    obs = student_term.year_of_enrollment_at_cohort_inst(
        df, cohort_start_dt_col=ccol, term_start_dt_col=tcol
    )
    assert isinstance(obs, pd.Series) and not obs.empty
    assert obs.equals(exp) or obs.compare(exp).empty


@pytest.mark.parametrize(
    ["df", "inst", "exp"],
    [
        (
            pd.DataFrame(
                {
                    "year_of_enrollment_at_cohort_inst": [1, 2, 3, 4],
                    "first_year_to_certificate_at_cohort_inst": [
                        2,
                        pd.NA,
                        2,
                        2,
                    ],
                    "years_to_latest_certificate_at_cohort_inst": [
                        3,
                        3,
                        pd.NA,
                        3,
                    ],
                    "first_year_to_certificate_at_other_inst": [
                        2,
                        pd.NA,
                        2,
                        2,
                    ],
                    "years_to_latest_certificate_at_other_inst": [
                        3,
                        3,
                        pd.NA,
                        3,
                    ],
                },
                dtype="Int8",
            ),
            "cohort",
            pd.Series([False, False, True, True], dtype="boolean"),
        ),
    ],
)
def test_student_earned_certificate(df, inst, exp):
    obs = student_term.student_earned_certificate(df, inst=inst)
    assert isinstance(obs, pd.Series) and not obs.empty
    assert pd.testing.assert_series_equal(obs, exp) is None


@pytest.mark.parametrize(
    ["df", "ccol", "tcol", "exp"],
    [
        (
            pd.DataFrame(
                {
                    "cohort_start_dt": ["2019-09-01", "2019-09-01", "2021-02-01"],
                    "term_start_dt": ["2020-02-01", "2019-09-01", "2020-09-01"],
                },
                dtype="datetime64[s]",
            ),
            "cohort_start_dt",
            "term_start_dt",
            pd.Series([False, False, True], dtype="boolean"),
        ),
    ],
)
def test_term_is_pre_cohort(df, ccol, tcol, exp):
    obs = student_term.term_is_pre_cohort(
        df, cohort_start_dt_col=ccol, term_start_dt_col=tcol
    )
    assert isinstance(obs, pd.Series) and not obs.empty
    assert obs.equals(exp) or obs.compare(exp).empty


@pytest.mark.parametrize(
    ["df", "study_area_col", "exp"],
    [
        (
            pd.DataFrame(
                {
                    "study_area_term_1": ["01", "02", None, "03"],
                    "study_area_year_1": ["01", "03", None, "03"],
                    "course_subject_areas": [
                        ["01", "01", "01", "02"],
                        ["01", "02", "01"],
                        ["01", "02", "03"],
                        [],
                    ],
                }
            ).astype({"study_area_term_1": "string", "study_area_year_1": "string"}),
            "study_area_term_1",
            pd.Series([3, 1, 0, 0], dtype="Int8"),
        ),
        (
            pd.DataFrame(
                {
                    "study_area_term_1": ["01", "02", None, "03"],
                    "study_area_year_1": ["01", "03", None, "03"],
                    "course_subject_areas": [
                        ["01", "01", "01", "02"],
                        ["01", "02", "01"],
                        ["01", "02", "03"],
                        [],
                    ],
                }
            ).astype({"study_area_term_1": "string", "study_area_year_1": "string"}),
            "study_area_year_1",
            pd.Series([3, 0, 0, 0], dtype="Int8"),
        ),
    ],
)
def test_num_courses_in_study_area(df, study_area_col, exp):
    obs = student_term.num_courses_in_study_area(df, study_area_col=study_area_col)
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


@pytest.mark.parametrize(
    ["df", "min_num_credits_full_time", "num_credits_col", "exp"],
    [
        (
            pd.DataFrame({"num_credits_attempted": [15.0, 12.0, 8.0, 0.0]}),
            12.0,
            "num_credits_attempted",
            pd.Series(["FULL-TIME", "FULL-TIME", "PART-TIME", "PART-TIME"]),
        ),
    ],
)
def test_student_term_enrollment_intensity(
    df, min_num_credits_full_time, num_credits_col, exp
):
    obs = student_term.student_term_enrollment_intensity(
        df,
        min_num_credits_full_time=min_num_credits_full_time,
        num_credits_col=num_credits_col,
    )
    assert isinstance(obs, pd.Series) and not obs.empty
    assert obs.equals(exp) or obs.compare(exp).empty
