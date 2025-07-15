import numpy as np
import pandas as pd
import pytest

from student_success_tool.preprocessing.features.pdp import cumulative


@pytest.fixture(scope="module")
def df():
    return pd.DataFrame(
        {
            "student_id": ["01", "01", "01", "01", "01"],
            "academic_year": [
                "2020-21",
                "2020-21",
                "2020-21",
                "2021-22",
                "2022-23",
            ],
            "academic_term": ["FALL", "SPRING", "SUMMER", "FALL", "FALL"],
            "term_id": [
                "2020-21 FALL",
                "2020-21 SPRING",
                "2020-21 SUMMER",
                "2021-22 FALL",
                "2022-23 FALL",
            ],
            "course_grade_num_mean": [4.0, 2.5, 1.75, 3.25, 3.5],
            "num_courses": [3, 2, 2, 2, 1],
            "num_courses_course_level_0": [2, 1, 0, 0, 0],
            "num_courses_course_level_1": [1, 1, 2, 2, 1],
            "took_course_subject_area_02": [True, False, False, False, True],
            "took_course_subject_area_05_06": [False, False, False, True, False],
            "num_credits_earned": [3, 12, 15, 4, 15],
            "course_ids": [
                ["A101", "B101", "C201"],
                ["A101", "D201"],
                ["E201", "F201"],
                ["G201", "H201"],
                ["H201"],
            ],
            "course_subject_areas": [
                ["01", "02", "03"],
                ["01", "04"],
                ["03", "04"],
                ["05", "06"],
                ["02"],
            ],
            "term_rank": [1, 3, 4, 5, 9],
            "term_rank_core": [1, 2, pd.NA, 3, 5],
            "term_rank_noncore": [pd.NA, pd.NA, 2, pd.NA, pd.NA],
            "term_is_pre_cohort": [True, False, False, False, False],
        }
    ).astype(
        {"term_rank": "Int8", "term_rank_core": "Int8", "term_rank_noncore": "Int8"}
    )


@pytest.fixture(scope="module")
def df_grped(df):
    return (
        df.sort_values(by=["student_id", "academic_year", "academic_term"])
        .groupby(by="student_id", as_index=True, observed=True, sort=False)
    )  # fmt: skip


@pytest.mark.parametrize(
    ["num_course_cols", "dummy_course_cols", "col_aggs", "exp"],
    [
        (
            [
                "num_courses_course_level_0",
                "num_courses_course_level_1",
                "num_credits_earned",
            ],
            ["took_course_subject_area_02", "took_course_subject_area_05_06"],
            [
                ("term_id", "count"),
                ("term_is_pre_cohort", "sum"),
                ("course_grade_num_mean", ["mean", "std"]),
                ("num_courses", "sum"),
            ],
            pd.DataFrame(
                {
                    "cumcount_term_id": [1.0, 2.0, 3.0, 4.0, 5.0],
                    "cumsum_term_is_pre_cohort": [1.0, 1.0, 1.0, 1.0, 1.0],
                    "cummean_course_grade_num_mean": [4.0, 3.25, 2.75, 2.875, 3.0],
                    "cumstd_course_grade_num_mean": [
                        np.nan,
                        1.061,
                        1.146,
                        0.968,
                        0.884,
                    ],
                    "cumsum_num_courses": [3.0, 5.0, 7.0, 9.0, 10.0],
                    "cumsum_num_credits_earned": [3.0, 15.0, 30.0, 34.0, 49.0],
                    "cummax_took_course_subject_area_02": [1.0, 1.0, 1.0, 1.0, 1.0],
                    "cummax_took_course_subject_area_05_06": [
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        1.0,
                    ],
                    "cumfrac_num_courses_course_level_0": [
                        0.667,
                        0.6,
                        0.429,
                        0.333,
                        0.3,
                    ],
                    "cumfrac_num_courses_course_level_1": [
                        0.333,
                        0.4,
                        0.571,
                        0.667,
                        0.7,
                    ],
                    "cummax_in_12_creds_took_course_subject_area_02": [
                        True,
                        True,
                        True,
                        True,
                        True,
                    ],
                    "cummax_in_12_creds_took_course_subject_area_05_06": [
                        False,
                        False,
                        False,
                        False,
                        False,
                    ],
                }
            ),
        ),
    ],
)
def test_expanding_agg_features(
    df_grped, num_course_cols, dummy_course_cols, col_aggs, exp
):
    obs = cumulative.expanding_agg_features(
        df_grped,
        num_course_cols=num_course_cols,
        col_aggs=col_aggs,
        dummy_course_cols=dummy_course_cols,
        credits=12,
    )
    assert isinstance(obs, pd.DataFrame) and not obs.empty
    # raises error if not equal
    print(obs.columns)
    print(exp.columns)
    assert pd.testing.assert_frame_equal(obs, exp, rtol=0.001) is None


@pytest.mark.parametrize(
    ["cols", "exp"],
    [
        (
            ["course_ids", "course_subject_areas"],
            pd.DataFrame(
                {
                    "cumnum_unique_course_ids": [3, 4, 6, 8, 8],
                    "cumnum_unique_course_subject_areas": [3, 4, 4, 6, 6],
                    "cumnum_repeated_course_ids": [0, 1, 1, 1, 2],
                    "cumnum_repeated_course_subject_areas": [0, 1, 3, 3, 4],
                },
                dtype="Int16",
            ),
        ),
    ],
)
def test_cumnum_unique_and_repeated_features(df_grped, cols, exp):
    obs = cumulative.cumnum_unique_and_repeated_features(df_grped, cols=cols)
    assert isinstance(obs, pd.DataFrame) and not obs.empty
    # raises error if not equal
    assert pd.testing.assert_frame_equal(obs, exp, rtol=0.001) is None


@pytest.mark.parametrize(
    "exp_new",
    [
        pd.DataFrame(
            {
                "min_student_term_rank": [1, 1, 1, 1, 1],
                "min_student_term_rank_core": [1, 1, 1, 1, 1],
                "min_student_term_rank_noncore": [2, 2, 2, 2, 2],
                "cumfrac_terms_enrolled": [1.0, 0.666, 0.75, 0.8, 0.556],
                "cumfrac_core_terms_enrolled": [1.0, 1.0, pd.NA, 1.0, 0.8],
                "cumfrac_noncore_terms_enrolled": [pd.NA, pd.NA, 1.0, pd.NA, pd.NA],
            }
        ).astype(
            {
                "min_student_term_rank": "Int8",
                "min_student_term_rank_core": "Int8",
                "min_student_term_rank_noncore": "Int8",
                "cumfrac_terms_enrolled": "Float32",
                "cumfrac_core_terms_enrolled": "Float32",
                "cumfrac_noncore_terms_enrolled": "Float32",
            }
        ),
    ],
)
def test_add_cumfrac_terms_enrolled_features(df, exp_new):
    # HACK: let's add the couple cumulative feature pre-requisites here
    df = df.assign(
        cumnum_terms_enrolled=pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]),
        cumnum_core_terms_enrolled=pd.Series([1.0, 2.0, 2.0, 3.0, 4.0]),
        cumnum_noncore_terms_enrolled=pd.Series([0.0, 0.0, 1.0, 1.0, 1.0]),
    )
    obs = cumulative.add_cumfrac_terms_enrolled_features(
        df, student_id_cols=["student_id"]
    )
    assert isinstance(obs, pd.DataFrame) and not obs.empty
    # all *existing* columns are unchanged
    assert obs.loc[:, df.columns].equals(df)
    # all *new* columns are as expected
    assert (
        pd.testing.assert_frame_equal(obs.drop(columns=df.columns), exp_new, rtol=0.01)
        is None
    )


@pytest.mark.parametrize(
    ["cols", "max_term_num", "exp_new"],
    [
        (
            ["num_courses", "course_grade_num_mean"],
            3,
            pd.DataFrame(
                {
                    "num_courses_diff_prev_term": [np.nan, -1.0, 0.0, 0.0, -1.0],
                    "course_grade_num_mean_diff_prev_term": [
                        np.nan,
                        -1.5,
                        -0.75,
                        1.5,
                        0.25,
                    ],
                    "num_courses_diff_term_1_to_term_2": [-1.0, -1.0, -1.0, -1.0, -1.0],
                    "num_courses_diff_term_2_to_term_3": [0.0, 0.0, 0.0, 0.0, 0.0],
                    "course_grade_num_mean_diff_term_1_to_term_2": [
                        -1.5,
                        -1.5,
                        -1.5,
                        -1.5,
                        -1.5,
                    ],
                    "course_grade_num_mean_diff_term_2_to_term_3": [
                        -0.75,
                        -0.75,
                        -0.75,
                        -0.75,
                        -0.75,
                    ],
                }
            ),
        ),
        (
            ["num_courses"],
            2,
            pd.DataFrame(
                {
                    "num_courses_diff_prev_term": [np.nan, -1.0, 0.0, 0.0, -1.0],
                    "num_courses_diff_term_1_to_term_2": [-1.0, -1.0, -1.0, -1.0, -1.0],
                }
            ),
        ),
    ],
)
def test_add_term_diff_features(df, cols, max_term_num, exp_new):
    # HACK: let's add the couple cumulative feature pre-requisites here
    df = df.assign(
        cumnum_terms_enrolled=pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]),
        cumnum_core_terms_enrolled=pd.Series([1.0, 2.0, 2.0, 3.0, 4.0]),
    )
    obs = cumulative.add_term_diff_features(
        df,
        cols=cols,
        max_term_num=max_term_num,
        student_id_cols=["student_id"],
        term_num_col="cumnum_terms_enrolled",
    )
    assert isinstance(obs, pd.DataFrame) and not obs.empty
    # all *existing* columns are unchanged
    assert obs.loc[:, df.columns].equals(df)
    # all *new* columns are as expected
    assert (
        pd.testing.assert_frame_equal(obs.drop(columns=df.columns), exp_new, rtol=0.01)
        is None
    )
