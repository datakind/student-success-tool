import pandas as pd
import pytest

from student_success_tool.preprocessing.selection import pdp


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
