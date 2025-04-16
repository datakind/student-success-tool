import functools as ft

import pandas as pd
import pytest

from student_success_tool.selection import pdp
from student_success_tool.targets.pdp import shared


@pytest.fixture(scope="module")
def df_test():
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
            pd.DataFrame(
                data={
                    "credential_sought": ["Associate's", "Associate's"],
                    "enrollment_type": ["FIRST-TIME", "FIRST-TIME"],
                    "enrollment_intensity": ["FULL-TIME", "PART-TIME"],
                },
                index=pd.Index(["01", "03"], name="student_id", dtype="string"),
            ).astype(
                {
                    "credential_sought": "string",
                    "enrollment_type": "string",
                    "enrollment_intensity": "string",
                }
            ),
        ),
        (
            {
                "enrollment_type": "FIRST-TIME",
                "enrollment_intensity": "PART-TIME",
            },
            pd.DataFrame(
                data={
                    "enrollment_type": ["FIRST-TIME", "FIRST-TIME"],
                    "enrollment_intensity": ["PART-TIME", "PART-TIME"],
                },
                index=pd.Index(["02", "03"], name="student_id", dtype="string"),
            ).astype({"enrollment_type": "string", "enrollment_intensity": "string"}),
        ),
        (
            {"credential_sought": ["Associate's", "Bachelor's"]},
            pd.DataFrame(
                data={
                    "credential_sought": [
                        "Associate's",
                        "Bachelor's",
                        "Associate's",
                        "Associate's",
                    ],
                },
                index=pd.Index(
                    ["01", "02", "03", "04"], name="student_id", dtype="string"
                ),
            ).astype({"credential_sought": "string"}),
        ),
        (
            {"enrollment_type": "RE-ADMIT"},
            pd.DataFrame(
                data={"enrollment_type": []},
                index=pd.Index([], name="student_id", dtype="string"),
            ).astype("string"),
        ),
    ],
)
def test_select_students_by_attributes(df_test, criteria, exp):
    obs = pdp.select_students_by_attributes(
        df_test, student_id_cols="student_id", **criteria
    )
    assert isinstance(obs, pd.DataFrame)
    assert pd.testing.assert_frame_equal(obs, exp) is None


@pytest.mark.parametrize(
    "exp",
    [
        pd.DataFrame(
            data={
                "student_cohort_year": [2020, 2021, 2019, 2020],
                "max_academic_year": [2022, 2022, 2022, 2022],
            },
            index=pd.Index(["01", "02", "03", "04"], name="student_id", dtype="string"),
            dtype="Int32",
        )
    ],
)
def test_select_students_by_second_year_data(df_test, exp):
    obs = pdp.select_students_by_second_year_data(
        df_test,
        student_id_cols="student_id",
        cohort_id_col="cohort_id",
        term_id_col="term_id",
    )
    assert isinstance(obs, pd.DataFrame)
    assert pd.testing.assert_frame_equal(obs, exp) is None


@pytest.mark.parametrize(
    ["checkpoint", "intensity_time_limits", "max_term_rank", "exp"],
    [
        (
            # first term as checkpoint
            pd.DataFrame(
                {
                    "student_id": ["01", "02", "03", "04", "05"],
                    "enrollment_intensity": [
                        "FULL-TIME",
                        "PART-TIME",
                        "PART-TIME",
                        "FULL-TIME",
                        pd.NA,
                    ],
                    "term_rank": [3, 5, 2, 4, 8],
                }
            ).astype({"student_id": "string", "enrollment_intensity": "string"}),
            {"FULL-TIME": (2, "term"), "PART-TIME": (4, "term")},
            "infer",
            pd.DataFrame(
                data={
                    "student_max_term_rank": [5, 6, 6],
                    "max_term_rank": [8, 8, 8],
                },
                index=pd.Index(["01", "03", "04"], dtype="string", name="student_id"),
            ).astype("Int8"),
        ),
        (
            # first term as checkpoint via callable
            ft.partial(
                shared.get_first_student_terms,
                student_id_cols="student_id",
                sort_cols="term_rank",
                include_cols=["enrollment_intensity"],
            ),
            {"FULL-TIME": (2, "term"), "PART-TIME": (4, "term")},
            "infer",
            pd.DataFrame(
                data={
                    "student_max_term_rank": [6, 5, 6],
                    "max_term_rank": [8, 8, 8],
                },
                index=pd.Index(["03", "01", "04"], dtype="string", name="student_id"),
            ).astype("Int8"),
        ),
        (
            # first term as checkpoint
            pd.DataFrame(
                {
                    "student_id": ["01", "02", "03", "04", "05"],
                    "enrollment_intensity": [
                        "FULL-TIME",
                        "PART-TIME",
                        "PART-TIME",
                        "FULL-TIME",
                        pd.NA,
                    ],
                    "term_rank": [3, 5, 2, 4, 8],
                }
            ).astype({"student_id": "string", "enrollment_intensity": "string"}),
            # same time limits for every intensity
            {"*": (3, "term")},
            # non-inferred max term rank
            7,
            pd.DataFrame(
                data={
                    "student_max_term_rank": [6, 5, 7],
                    "max_term_rank": [7, 7, 7],
                },
                index=pd.Index(["01", "03", "04"], dtype="string", name="student_id"),
            ).astype("Int8"),
        ),
        (
            # first term as checkpoint
            pd.DataFrame(
                {
                    "student_id": ["01", "02", "03", "04", "05"],
                    "enrollment_intensity": [
                        "FULL-TIME",
                        "PART-TIME",
                        "PART-TIME",
                        "FULL-TIME",
                        pd.NA,
                    ],
                    "term_rank": [3, 5, 2, 4, 8],
                }
            ).astype({"student_id": "string", "enrollment_intensity": "string"}),
            # time limits for full-time only, in years
            {"FULL-TIME": (1, "year")},
            "infer",
            pd.DataFrame(
                data={
                    "student_max_term_rank": [7, 8],
                    "max_term_rank": [8, 8],
                },
                index=pd.Index(["01", "04"], dtype="string", name="student_id"),
            ).astype("Int8"),
        ),
    ],
)
def test_select_students_with_max_target_term_in_dataset(
    df_test, checkpoint, intensity_time_limits, max_term_rank, exp
):
    obs = pdp.select_students_with_max_target_term_in_dataset(
        df_test,
        checkpoint=checkpoint,
        intensity_time_limits=intensity_time_limits,
        max_term_rank=max_term_rank,
        student_id_cols="student_id",
        enrollment_intensity_col="enrollment_intensity",
        term_rank_col="term_rank",
    )
    assert isinstance(obs, pd.DataFrame)
    assert pd.testing.assert_frame_equal(obs, exp) is None
