import pandas as pd
import pytest

from student_success_tool.checkpoints import pdp


@pytest.fixture(scope="module")
def df_test():
    return pd.DataFrame(
        {
            "student_id": ["01", "01", "01", "02", "02", "03", "04", "05"],
            "term_rank": [3, 4, 5, 5, 6, 2, 4, 8],
            "cohort_id": [
                "2020-21 SPRING",
                "2020-21 SPRING",
                "2020-21 SPRING",
                "2021-22 FALL",
                "2021-22 FALL",
                "2019-20 FALL",
                "2020-21 SPRING",
                "2022-23 FALL",
            ],
            "term_id": [
                "2020-21 FALL",
                "2020-21 SPRING",
                "2021-22 FALL",
                "2021-22 FALL",
                "2023-24 FALL",
                "2019-20 SPRING",
                "2020-21 SPRING",
                "2022-23 FALL",
            ],
            "enrollment_year": [1, 1, 2, 1, 3, 1, 1, 1],
            "enrollment_intensity": [
                "FULL-TIME",
                "FULL-TIME",
                "FULL-TIME",
                "PART-TIME",
                "PART-TIME",
                "PART-TIME",
                "FULL-TIME",
                pd.NA,
            ],
            "num_credits_earned": [25.0, 30.0, 35.0, 25.0, 35.0, 20.0, 45.0, 10.0],
            "term_is_pre_cohort": [
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        },
    ).astype(
        {
            "student_id": "string",
            "cohort_id": "string",
            "term_id": "string",
            "enrollment_intensity": "string",
        }
    )


@pytest.mark.parametrize(
    ["n", "include_cols", "exp"],
    [
        (
            0,
            None,
            pd.DataFrame(
                data={
                    "student_id": ["01", "02", "03", "04", "05"],
                    "term_rank": [3, 5, 2, 4, 8],
                    "cohort_id": [
                        "2020-21 SPRING",
                        "2021-22 FALL",
                        "2019-20 FALL",
                        "2020-21 SPRING",
                        "2022-23 FALL",
                    ],
                    "term_id": [
                        "2020-21 FALL",
                        "2021-22 FALL",
                        "2019-20 SPRING",
                        "2020-21 SPRING",
                        "2022-23 FALL",
                    ],
                    "enrollment_year": [1, 1, 1, 1, 1],
                    "enrollment_intensity": [
                        "FULL-TIME",
                        "PART-TIME",
                        "PART-TIME",
                        "FULL-TIME",
                        pd.NA,
                    ],
                    "num_credits_earned": [25.0, 25.0, 20.0, 45.0, 10.0],
                    "term_is_pre_cohort": [True, False, False, False, False],
                },
                index=pd.Index([0, 3, 5, 6, 7], dtype="int64"),
            ).astype(
                {
                    "student_id": "string",
                    "cohort_id": "string",
                    "term_id": "string",
                    "enrollment_intensity": "string",
                }
            ),
        ),
        (
            0,
            ["term_id"],
            pd.DataFrame(
                data={
                    "student_id": ["01", "02", "03", "04", "05"],
                    "term_rank": [3, 5, 2, 4, 8],
                    "term_id": [
                        "2020-21 FALL",
                        "2021-22 FALL",
                        "2019-20 SPRING",
                        "2020-21 SPRING",
                        "2022-23 FALL",
                    ],
                },
                index=pd.Index([0, 3, 5, 6, 7], dtype="int64"),
            ).astype({"student_id": "string", "term_id": "string"}),
        ),
        (
            1,
            None,
            pd.DataFrame(
                data={
                    "student_id": ["01", "02"],
                    "term_rank": [4, 6],
                    "cohort_id": ["2020-21 SPRING", "2021-22 FALL"],
                    "term_id": ["2020-21 SPRING", "2023-24 FALL"],
                    "enrollment_year": [1, 3],
                    "enrollment_intensity": ["FULL-TIME", "PART-TIME"],
                    "num_credits_earned": [30.0, 35.0],
                    "term_is_pre_cohort": [False, False],
                },
                index=pd.Index([1, 4], dtype="int64"),
            ).astype(
                {
                    "student_id": "string",
                    "cohort_id": "string",
                    "term_id": "string",
                    "enrollment_intensity": "string",
                }
            ),
        ),
    ],
)
def test_nth_student_terms(df_test, n, include_cols, exp):
    obs = pdp.nth_student_terms(
        df_test,
        n=n,
        student_id_cols="student_id",
        sort_cols="term_rank",
        include_cols=include_cols,
    )
    assert isinstance(obs, pd.DataFrame)
    assert pd.testing.assert_frame_equal(obs, exp) is None
