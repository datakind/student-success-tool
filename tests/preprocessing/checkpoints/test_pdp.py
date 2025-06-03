import pandas as pd
import pytest

from student_success_tool.preprocessing.checkpoints import pdp


@pytest.fixture(scope="module")
def df_test():
    return pd.DataFrame(
        {
            "student_id": [
                "01",
                "01",
                "01",
                "01",
                "02",
                "02",
                "02",
                "03",
                "03",
                "04",
                "05",
            ],
            "term_rank": [3, 4, 5, 6, 5, 6, 7, 2, 3, 4, 8],
            "cohort_id": [
                "2020-21 SPRING",
                "2020-21 SPRING",
                "2020-21 SPRING",
                "2020-21 SPRING",
                "2021-22 FALL",
                "2021-22 FALL",
                "2021-22 FALL",
                "2019-20 FALL",
                "2019-20 FALL",
                "2020-21 SPRING",
                "2022-23 FALL",
            ],
            "term_id": [
                "2020-21 FALL",
                "2020-21 SPRING",
                "2021-22 FALL",
                "2021-22 WINTER",
                "2021-22 FALL",
                "2023-24 FALL",
                "2024-25 SUMMER",
                "2019-20 SPRING",
                "2019-20 WINTER",
                "2020-21 SPRING",
                "2022-23 FALL",
            ],
            "enrollment_year": [1, 1, 2, 2, 1, 3, 4, 1, 1, 1, 1],
            "enrollment_intensity": [
                "FULL-TIME",
                "FULL-TIME",
                "FULL-TIME",
                "FULL-TIME",
                "PART-TIME",
                "PART-TIME",
                "PART-TIME",
                "PART-TIME",
                "PART-TIME",
                "FULL-TIME",
                pd.NA,
            ],
            "num_credits_earned": [
                25.0,
                30.0,
                35.0,
                12.0,
                25.0,
                35.0,
                18.0,
                20.0,
                5.0,
                45.0,
                10.0,
            ],
            "term_is_pre_cohort": [
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
            "term_is_core": [
                True,
                True,
                True,
                False,
                True,
                True,
                False,
                True,
                False,
                True,
                True,
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
    [
        "n",
        "include_cols",
        "exclude_pre_cohort_terms",
        "exclude_non_core_terms",
        "enrollment_year_col",
        "valid_enrollment_year",
        "exp",
    ],
    [
        (
            0,
            None,
            False,
            False,
            "enrollment_year",
            1,
            pd.DataFrame(
                {
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
                    "term_is_core": [True, True, True, True, True],
                }
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
            True,
            True,
            None,
            None,
            pd.DataFrame(
                {
                    "student_id": ["01", "02", "03", "04", "05"],
                    "term_rank": [4, 5, 2, 4, 8],
                    "term_id": [
                        "2020-21 SPRING",
                        "2021-22 FALL",
                        "2019-20 SPRING",
                        "2020-21 SPRING",
                        "2022-23 FALL",
                    ],
                }
            ).astype({"student_id": "string", "term_id": "string"}),
        ),
        (
            1,
            None,
            True,
            True,
            "enrollment_year",
            3,
            pd.DataFrame(
                {
                    "student_id": ["02"],
                    "term_rank": [6],
                    "cohort_id": ["2021-22 FALL"],
                    "term_id": ["2023-24 FALL"],
                    "enrollment_year": [3],
                    "enrollment_intensity": ["PART-TIME"],
                    "num_credits_earned": [35.0],
                    "term_is_pre_cohort": [False],
                    "term_is_core": [True],
                }
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
            -1,
            None,
            False,
            False,
            None,
            None,
            pd.DataFrame(
                {
                    "student_id": ["01", "02", "03", "04", "05"],
                    "term_rank": [6, 7, 3, 4, 8],
                    "cohort_id": [
                        "2020-21 SPRING",
                        "2021-22 FALL",
                        "2019-20 FALL",
                        "2020-21 SPRING",
                        "2022-23 FALL",
                    ],
                    "term_id": [
                        "2021-22 WINTER",
                        "2024-25 SUMMER",
                        "2019-20 WINTER",
                        "2020-21 SPRING",
                        "2022-23 FALL",
                    ],
                    "enrollment_year": [2, 4, 1, 1, 1],
                    "enrollment_intensity": [
                        "FULL-TIME",
                        "PART-TIME",
                        "PART-TIME",
                        "FULL-TIME",
                        pd.NA,
                    ],
                    "num_credits_earned": [12.0, 18.0, 5.0, 45.0, 10.0],
                    "term_is_pre_cohort": [False, False, False, False, False],
                    "term_is_core": [False, False, False, True, True],
                }
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
def test_nth_student_terms(
    df_test,
    n,
    include_cols,
    exclude_pre_cohort_terms,
    exclude_non_core_terms,
    enrollment_year_col,
    valid_enrollment_year,
    exp,
):
    from student_success_tool.preprocessing.checkpoints import pdp

    obs = pdp.nth_student_terms(
        df_test,
        n=n,
        student_id_cols="student_id",
        sort_cols="term_rank",
        include_cols=include_cols,
        term_is_pre_cohort_col="term_is_pre_cohort",
        exclude_pre_cohort_terms=exclude_pre_cohort_terms,
        term_is_core_col="term_is_core",
        exclude_non_core_terms=exclude_non_core_terms,
        enrollment_year_col=enrollment_year_col,
        valid_enrollment_year=valid_enrollment_year,
    )
    assert isinstance(obs, pd.DataFrame)
    assert (
        pd.testing.assert_frame_equal(
            obs.sort_values("student_id").reset_index(drop=True),
            exp.sort_values("student_id").reset_index(drop=True),
        )
        is None
    )


@pytest.mark.parametrize(
    ["include_cols", "exclude_pre_cohort_terms", "exclude_non_core_terms", "exp"],
    [
        (
            ["term_id"],
            False,
            True,
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
    ],
)
def test_first_student_terms(
    df_test, include_cols, exclude_pre_cohort_terms, exclude_non_core_terms, exp
):
    obs = pdp.first_student_terms(
        df_test,
        student_id_cols="student_id",
        sort_cols="term_rank",
        include_cols=include_cols,
        term_is_pre_cohort_col="term_is_pre_cohort",
        exclude_pre_cohort_terms=exclude_pre_cohort_terms,
        term_is_core_col="term_is_core",
        exclude_non_core_terms=exclude_non_core_terms,
    )
    assert isinstance(obs, pd.DataFrame)
    pd.testing.assert_frame_equal(
        obs.reset_index(drop=True), exp.reset_index(drop=True)
    )


@pytest.mark.parametrize(
    ["include_cols", "exclude_pre_cohort_terms", "exclude_non_core_terms", "exp"],
    [
        (
            ["term_id"],
            False,
            False,
            pd.DataFrame(
                data={
                    "student_id": ["01", "02", "03", "04", "05"],
                    "term_rank": [6, 7, 3, 4, 8],
                    "term_id": [
                        "2021-22 WINTER",
                        "2024-25 SUMMER",
                        "2019-20 WINTER",
                        "2020-21 SPRING",
                        "2022-23 FALL",
                    ],
                },
                index=pd.Index([2, 4, 5, 6, 7], dtype="int64"),
            ).astype({"student_id": "string", "term_id": "string"}),
        ),
    ],
)
def test_last_student_terms(
    df_test, include_cols, exclude_pre_cohort_terms, exclude_non_core_terms, exp
):
    obs = pdp.last_student_terms(
        df_test,
        student_id_cols="student_id",
        sort_cols="term_rank",
        include_cols=include_cols,
        term_is_pre_cohort_col="term_is_pre_cohort",
        exclude_pre_cohort_terms=exclude_pre_cohort_terms,
        term_is_core_col="term_is_core",
        exclude_non_core_terms=exclude_non_core_terms,
    )
    assert isinstance(obs, pd.DataFrame)
    pd.testing.assert_frame_equal(
        obs.reset_index(drop=True), exp.reset_index(drop=True)
    )


@pytest.mark.parametrize(
    ["min_num_credits", "include_cols", "exp"],
    [
        (
            30.0,
            ["num_credits_earned"],
            pd.DataFrame(
                data={
                    "student_id": ["01", "02", "04"],
                    "term_rank": [4, 6, 4],
                    "num_credits_earned": [30.0, 35.0, 45.0],
                },
                index=pd.Index([1, 4, 6], dtype="int64"),
            ).astype({"student_id": "string"}),
        ),
        (
            45.0,
            ["num_credits_earned"],
            pd.DataFrame(
                data={
                    "student_id": ["04"],
                    "term_rank": [4],
                    "num_credits_earned": [45.0],
                },
                index=pd.Index([6], dtype="int64"),
            ).astype({"student_id": "string"}),
        ),
    ],
)
def test_first_student_terms_at_num_credits_earned(
    df_test, min_num_credits, include_cols, exp
):
    obs = pdp.first_student_terms_at_num_credits_earned(
        df_test,
        min_num_credits=min_num_credits,
        num_credits_col="num_credits_earned",
        student_id_cols="student_id",
        sort_cols="term_rank",
        include_cols=include_cols,
    )
    assert isinstance(obs, pd.DataFrame)
    pd.testing.assert_frame_equal(
        obs.reset_index(drop=True), exp.reset_index(drop=True)
    )


@pytest.mark.parametrize(
    ["include_cols", "exp"],
    [
        (
            ["term_is_pre_cohort"],
            pd.DataFrame(
                data={
                    "student_id": ["01", "02", "03", "04", "05"],
                    "term_rank": [4, 5, 2, 4, 8],
                    "term_is_pre_cohort": [False, False, False, False, False],
                },
                index=pd.Index([1, 3, 5, 6, 7], dtype="int64"),
            ).astype({"student_id": "string"}),
        ),
    ],
)
def test_first_student_terms_within_cohort(df_test, include_cols, exp):
    obs = pdp.first_student_terms_within_cohort(
        df_test,
        term_is_pre_cohort_col="term_is_pre_cohort",
        student_id_cols="student_id",
        sort_cols="term_rank",
        include_cols=include_cols,
    )
    assert isinstance(obs, pd.DataFrame)
    pd.testing.assert_frame_equal(
        obs.reset_index(drop=True), exp.reset_index(drop=True)
    )


@pytest.mark.parametrize(
    ["enrollment_year", "include_cols", "exp"],
    [
        (
            1,
            ["enrollment_year"],
            pd.DataFrame(
                data={
                    "student_id": ["01", "02", "03", "04", "05"],
                    "term_rank": [4, 5, 3, 4, 8],
                    "enrollment_year": [1, 1, 1, 1, 1],
                },
                index=pd.Index([1, 4, 8, 9, 10], dtype="int64"),
            ).astype({"student_id": "string"}),
        ),
        (
            2,
            ["enrollment_year"],
            pd.DataFrame(
                data={
                    "student_id": ["01"],
                    "term_rank": [6],
                    "enrollment_year": [2],
                },
                index=pd.Index([3], dtype="int64"),
            ).astype({"student_id": "string"}),
        ),
    ],
)
def test_last_student_terms_in_enrollment_year(
    df_test, enrollment_year, include_cols, exp
):
    obs = pdp.last_student_terms_in_enrollment_year(
        df_test,
        enrollment_year=enrollment_year,
        enrollment_year_col="enrollment_year",
        student_id_cols="student_id",
        sort_cols="term_rank",
        include_cols=include_cols,
    )
    assert isinstance(obs, pd.DataFrame)
    pd.testing.assert_frame_equal(
        obs.reset_index(drop=True), exp.reset_index(drop=True)
    )
