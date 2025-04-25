import pandas as pd
import pytest

from student_success_tool.targets.pdp import shared


@pytest.fixture(scope="module")
def test_df():
    return pd.DataFrame(
        {
            "student_id": ["01", "01", "01", "02", "02", "03", "04", "05"],
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
                "2021-22 SPRING",
                "2019-20 SPRING",
                "2020-21 SPRING",
                "2022-23 FALL",
            ],
            "credential_sought": [
                "Associate's",
                "Associate's",
                "Associate's",
                "Bachelor's",
                "Bachelor's",
                "Associate's",
                "Associate's",
                pd.NA,
            ],
            "enrollment_type": [
                "FIRST-TIME",
                "FIRST-TIME",
                "FIRST-TIME",
                "FIRST-TIME",
                "FIRST-TIME",
                "FIRST-TIME",
                "TRANSFER-IN",
                pd.NA,
            ],
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
            "term_rank": [3, 4, 5, 5, 6, 2, 4, 8],
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
            "credential_sought": "string",
            "enrollment_type": "string",
            "enrollment_intensity": "string",
        }
    )


@pytest.mark.parametrize(
    ["criteria", "exp"],
    [
        (
            {
                "credential_sought": "Associate's",
                "enrollment_type": "FIRST-TIME",
                "enrollment_intensity": {"FULL-TIME", "PART-TIME"},
            },
            pd.DataFrame({"student_id": ["01", "03"]}),
        ),
        (
            {
                "enrollment_type": "FIRST-TIME",
                "enrollment_intensity": "PART-TIME",
            },
            pd.DataFrame({"student_id": ["02", "03"]}),
        ),
        (
            {"credential_sought": ["Associate's", "Bachelor's"]},
            pd.DataFrame({"student_id": ["01", "02", "03", "04"]}),
        ),
        (
            {"enrollment_type": "RE-ADMIT"},
            pd.DataFrame({"student_id": []}),
        ),
    ],
)
def test_select_students_by_criteria(test_df, criteria, exp):
    obs = shared.select_students_by_criteria(
        test_df, student_id_cols="student_id", **criteria
    )
    assert isinstance(obs, pd.DataFrame)
    assert obs.equals(exp) or obs.compare(exp).empty


@pytest.mark.parametrize(
    ["intensity_time_lefts", "max_term_rank", "num_terms_in_year", "exp"],
    [
        (
            [("FULL-TIME", 1.0, "year"), ("PART-TIME", 2.0, "year")],
            8,
            4,
            pd.DataFrame({"student_id": ["01", "04"]}),
        ),
        (
            [("FULL-TIME", 4.0, "term"), ("PART-TIME", 8.0, "term")],
            8,
            4,
            pd.DataFrame({"student_id": ["01", "04"]}),
        ),
        (
            [("FULL-TIME", 1.0, "year"), ("PART-TIME", 2.0, "year")],
            10,
            4,
            pd.DataFrame({"student_id": ["01", "03", "04"]}),
        ),
        (
            [("FULL-TIME", 1.0, "year")],
            8,
            4,
            pd.DataFrame({"student_id": ["01", "04"]}),
        ),
        (
            [("PART-TIME", 2.0, "year")],
            8,
            4,
            pd.DataFrame({"student_id": []}),
        ),
        (
            [("FULL-TIME", 1.0, "year"), ("PART-TIME", 2.0, "year")],
            8,
            3,
            pd.DataFrame({"student_id": ["01", "03", "04"]}),
        ),
    ],
)
def test_select_students_by_time_left(
    test_df, intensity_time_lefts, max_term_rank, num_terms_in_year, exp
):
    obs = shared.select_students_by_time_left(
        test_df,
        intensity_time_lefts=intensity_time_lefts,
        max_term_rank=max_term_rank,
        num_terms_in_year=num_terms_in_year,
        student_id_cols="student_id",
        enrollment_intensity_col="enrollment_intensity",
        term_rank_col="term_rank",
    )
    assert isinstance(obs, pd.DataFrame)
    assert obs.equals(exp) or obs.compare(exp).empty


@pytest.mark.parametrize(
    "exp",
    [
        pd.DataFrame({"student_id": ["01", "02", "03", "04"]}),
    ],
)
def test_select_students_by_next_year_course_data(test_df, exp):
    obs = shared.select_students_by_next_year_course_data(
        test_df, student_id_cols="student_id"
    )
    assert isinstance(obs, pd.DataFrame)
    assert obs.equals(exp) or obs.compare(exp).empty


@pytest.mark.parametrize(
    ["include_cols", "exp"],
    [
        (
            [],
            pd.DataFrame(
                {
                    "student_id": ["03", "01", "04", "02", "05"],
                    "term_rank": [2, 3, 4, 5, 8],
                }
            ),
        ),
        (
            ["num_credits_earned"],
            pd.DataFrame(
                {
                    "student_id": ["03", "01", "04", "02", "05"],
                    "term_rank": [2, 3, 4, 5, 8],
                    "num_credits_earned": [20.0, 25.0, 45.0, 25.0, 10.0],
                }
            ),
        ),
        (
            ["num_credits_earned", "term_rank", "student_id"],
            pd.DataFrame(
                {
                    "student_id": ["03", "01", "04", "02", "05"],
                    "term_rank": [2, 3, 4, 5, 8],
                    "num_credits_earned": [20.0, 25.0, 45.0, 25.0, 10.0],
                }
            ),
        ),
        (
            None,
            pd.DataFrame(
                {
                    "student_id": ["03", "01", "04", "02", "05"],
                    "cohort_id": [
                        "2019-20 FALL",
                        "2020-21 SPRING",
                        "2020-21 SPRING",
                        "2021-22 FALL",
                        "2022-23 FALL",
                    ],
                    "term_id": [
                        "2019-20 SPRING",
                        "2020-21 FALL",
                        "2020-21 SPRING",
                        "2021-22 FALL",
                        "2022-23 FALL",
                    ],
                    "credential_sought": [
                        "Associate's",
                        "Associate's",
                        "Associate's",
                        "Bachelor's",
                        pd.NA,
                    ],
                    "enrollment_type": [
                        "FIRST-TIME",
                        "FIRST-TIME",
                        "TRANSFER-IN",
                        "FIRST-TIME",
                        pd.NA,
                    ],
                    "enrollment_intensity": [
                        "PART-TIME",
                        "FULL-TIME",
                        "FULL-TIME",
                        "PART-TIME",
                        pd.NA,
                    ],
                    "num_credits_earned": [20.0, 25.0, 45.0, 25.0, 10.0],
                    "term_rank": [2, 3, 4, 5, 8],
                    "term_is_pre_cohort": [False, True, False, False, False],
                }
            ).astype(
                {
                    "credential_sought": "string",
                    "enrollment_type": "string",
                    "enrollment_intensity": "string",
                }
            ),
        ),
    ],
)
def test_get_first_student_terms(test_df, include_cols, exp):
    obs = shared.get_first_student_terms(
        test_df,
        student_id_cols="student_id",
        sort_cols="term_rank",
        include_cols=include_cols,
    )
    assert isinstance(obs, pd.DataFrame)
    assert obs.equals(exp) or obs.compare(exp).empty


@pytest.mark.parametrize(
    ["min_num_credits", "include_cols", "exp"],
    [
        (
            10.0,
            [],
            pd.DataFrame(
                {
                    "student_id": ["03", "01", "04", "02", "05"],
                    "term_rank": [2, 3, 4, 5, 8],
                }
            ),
        ),
        (
            30.0,
            ["num_credits_earned"],
            pd.DataFrame(
                {
                    "student_id": ["01", "04", "02"],
                    "term_rank": [4, 4, 6],
                    "num_credits_earned": [30.0, 45.0, 35.0],
                }
            ),
        ),
        (
            60.0,
            [],
            pd.DataFrame({"student_id": [], "term_rank": []}),
        ),
    ],
)
def test_get_first_student_terms_at_num_credits_earned(
    test_df, min_num_credits, include_cols, exp
):
    obs = shared.get_first_student_terms_at_num_credits_earned(
        test_df,
        min_num_credits=min_num_credits,
        student_id_cols="student_id",
        sort_cols="term_rank",
        num_credits_col="num_credits_earned",
        include_cols=include_cols,
    )
    assert isinstance(obs, pd.DataFrame)
    assert obs.equals(exp) or obs.compare(exp).empty


@pytest.mark.parametrize(
    ["include_cols", "exp"],
    [
        (
            [],
            pd.DataFrame(
                {
                    "student_id": ["03", "01", "04", "02", "05"],
                    "term_rank": [2, 4, 4, 5, 8],
                }
            ),
        ),
        (
            ["cohort_id", "term_id"],
            pd.DataFrame(
                {
                    "student_id": ["03", "01", "04", "02", "05"],
                    "term_rank": [2, 4, 4, 5, 8],
                    "cohort_id": [
                        "2019-20 FALL",
                        "2020-21 SPRING",
                        "2020-21 SPRING",
                        "2021-22 FALL",
                        "2022-23 FALL",
                    ],
                    "term_id": [
                        "2019-20 SPRING",
                        "2020-21 SPRING",
                        "2020-21 SPRING",
                        "2021-22 FALL",
                        "2022-23 FALL",
                    ],
                }
            ),
        ),
    ],
)
def test_get_first_student_terms_within_cohort(test_df, include_cols, exp):
    obs = shared.get_first_student_terms_within_cohort(
        test_df,
        student_id_cols="student_id",
        sort_cols="term_rank",
        include_cols=include_cols,
    )
    assert isinstance(obs, pd.DataFrame)
    assert obs.equals(exp) or obs.compare(exp).empty
