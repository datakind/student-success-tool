import numpy as np
import pandas as pd
import pytest

from student_success_tool.analysis.pdp.features import cumulative


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
                "2021-22",
            ],
            "academic_term": ["FALL", "SPRING", "SUMMER", "FALL", "SPRING"],
            "term_id": [
                "2020-21 FALL",
                "2020-21 SPRING",
                "2020-21 SUMMER",
                "2021-22 FALL",
                "2021-22 SPRING",
            ],
            "course_grade_num_mean": [4.0, 2.5, 1.75, 3.25, 3.5],
            "num_courses": [3, 2, 2, 2, 1],
            "num_courses_course_level_0": [2, 1, 0, 0, 0],
            "num_courses_course_level_1": [1, 1, 2, 2, 1],
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
        }
    )


@pytest.fixture(scope="module")
def df_grped(df):
    return (
        df.sort_values(by=["student_id", "academic_year", "academic_term"])
        .groupby(by="student_id", as_index=True, observed=True, sort=False)
    )  # fmt: skip


@pytest.mark.parametrize(
    ["num_course_cols", "col_aggs", "exp"],
    [
        (
            ["num_courses_course_level_0", "num_courses_course_level_1"],
            [
                ("term_id", "count"),
                ("course_grade_num_mean", ["mean", "std"]),
                ("num_courses", "sum"),
            ],
            pd.DataFrame(
                {
                    "term_id_cumcount": [1.0, 2.0, 3.0, 4.0, 5.0],
                    "course_grade_num_mean_cummean": [4.0, 3.25, 2.75, 2.875, 3.0],
                    "course_grade_num_mean_cumstd": [
                        np.nan,
                        1.061,
                        1.146,
                        0.968,
                        0.884,
                    ],
                    "num_courses_cumsum": [3.0, 5.0, 7.0, 9.0, 10.0],
                    "num_courses_course_level_0_cumfrac": [
                        0.667,
                        0.6,
                        0.429,
                        0.333,
                        0.3,
                    ],
                    "num_courses_course_level_1_cumfrac": [
                        0.333,
                        0.4,
                        0.571,
                        0.667,
                        0.7,
                    ],
                }
            ),
        ),
    ],
)
def test_expanding_agg_features(df_grped, num_course_cols, col_aggs, exp):
    obs = cumulative.expanding_agg_features(
        df_grped, num_course_cols=num_course_cols, col_aggs=col_aggs
    )
    assert isinstance(obs, pd.DataFrame) and not obs.empty
    # raises error if not equal
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
