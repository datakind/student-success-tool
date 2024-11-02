import numpy as np
import pandas as pd
import pytest

from student_success_tool.analysis.pdp.features import cumulative


@pytest.fixture(scope="module")
def df_grped():
    return (
        pd.DataFrame(
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
            }
        )
        .sort_values(by=["student_id", "academic_year", "academic_term"])
        .groupby(by="student_id", as_index=True, observed=True, sort=False)
    )


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
