import pandas as pd
import pytest

from student_success_tool.preprocessing import checkpoints
from student_success_tool.preprocessing.targets.pdp import shared


@pytest.fixture(scope="module")
def df_test_max_term():
    return pd.DataFrame(
        data={
            "student_id": ["01", "01", "02", "02", "03"],
            "enrollment_intensity": ["FT", "FT", "PT", "PT", "FT"],
            "term_rank": [1, 2, 4, 5, 10],
            "term_is_pre_cohort": [False, False, True, False, False],
        },
    ).astype({"student_id": "string", "enrollment_intensity": "string"})


@pytest.fixture(scope="module")
def df_test_year2():
    return pd.DataFrame(
        {
            "student_id": ["01", "01", "02", "03"],
            "cohort_id": [
                "2020-21 FALL",
                "2020-21 FALL",
                "2021-22 SPRING",
                "2022-23 FALL",
            ],
            "term_id": [
                "2020-21 FALL",
                "2020-21 SPRING",
                "2021-22 SPRING",
                "2022-23 FALL",
            ],
        },
        dtype="string",
    )


@pytest.mark.parametrize(
    [
        "checkpoint",
        "intensity_time_limits",
        "max_term_rank",
        "num_terms_in_year",
        "exp",
    ],
    [
        # max target term inferred
        (
            checkpoints.pdp.first_student_terms,
            {"FT": (1, "year"), "PT": (2, "year")},
            "infer",
            3,
            pd.DataFrame(
                data={"student_max_term_rank": [4, 10], "max_term_rank": [10, 10]},
                index=pd.Index(["01", "02"], name="student_id", dtype="string"),
            ).astype("Int8"),
        ),
        # max target term manually specified
        (
            checkpoints.pdp.first_student_terms,
            {"FT": (1, "year"), "PT": (2, "year")},
            13,
            3,
            pd.DataFrame(
                data={
                    "student_max_term_rank": [4, 10, 13],
                    "max_term_rank": [13, 13, 13],
                },
                index=pd.Index(["01", "02", "03"], name="student_id", dtype="string"),
            ).astype("Int8"),
        ),
        # num terms in year adjusted
        (
            checkpoints.pdp.first_student_terms,
            {"FT": (1, "year"), "PT": (2, "year")},
            "infer",
            4,
            pd.DataFrame(
                data={"student_max_term_rank": [5], "max_term_rank": [10]},
                index=pd.Index(["01"], name="student_id", dtype="string"),
            ).astype("Int8"),
        ),
        # checkpoint given as dataframe
        (
            pd.DataFrame(
                data={
                    "student_id": ["01", "02", "03"],
                    "enrollment_intensity": ["FT", "PT", "FT"],
                    "term_rank": [1, 4, 10],
                    "term_is_pre_cohort": [False, True, False],
                },
            ).astype({"student_id": "string", "enrollment_intensity": "string"}),
            {"FT": (1, "year"), "PT": (2, "year")},
            13,
            3,
            pd.DataFrame(
                data={
                    "student_max_term_rank": [4, 10, 13],
                    "max_term_rank": [13, 13, 13],
                },
                index=pd.Index(["01", "02", "03"], name="student_id", dtype="string"),
            ).astype("Int8"),
        ),
        # pre-cohort terms excluded when computing max target term
        (
            checkpoints.pdp.first_student_terms_within_cohort,
            {"FT": (1, "year"), "PT": (2, "year")},
            12,
            3,
            pd.DataFrame(
                data={"student_max_term_rank": [4, 11], "max_term_rank": [12, 12]},
                index=pd.Index(["01", "02"], name="student_id", dtype="string"),
            ).astype("Int8"),
        ),
    ],
)
def test_get_students_with_max_target_term_in_dataset(
    df_test_max_term,
    checkpoint,
    intensity_time_limits,
    max_term_rank,
    num_terms_in_year,
    exp,
):
    obs = shared.get_students_with_max_target_term_in_dataset(
        df_test_max_term,
        checkpoint=checkpoint,
        intensity_time_limits=intensity_time_limits,
        max_term_rank=max_term_rank,
        num_terms_in_year=num_terms_in_year,
        student_id_cols="student_id",
        enrollment_intensity_col="enrollment_intensity",
        term_rank_col="term_rank",
    )
    assert isinstance(obs, pd.DataFrame)
    assert pd.testing.assert_frame_equal(obs, exp) is None


@pytest.mark.parametrize(
    ["max_academic_year", "exp"],
    [
        (
            "infer",
            pd.DataFrame(
                data={
                    "student_cohort_year": [2020, 2021],
                    "max_academic_year": [2022, 2022],
                },
                index=pd.Index(["01", "02"], name="student_id", dtype="string"),
                dtype="Int32",
            ),
        ),
        (
            "2024-25",
            pd.DataFrame(
                data={
                    "student_cohort_year": [2020, 2021, 2022],
                    "max_academic_year": [2024, 2024, 2024],
                },
                index=pd.Index(["01", "02", "03"], name="student_id", dtype="string"),
                dtype="Int32",
            ),
        ),
    ],
)
def test_get_students_with_second_year_in_dataset(
    df_test_year2, max_academic_year, exp
):
    obs = shared.get_students_with_second_year_in_dataset(
        df_test_year2,
        max_academic_year=max_academic_year,
        student_id_cols="student_id",
        cohort_id_col="cohort_id",
        term_id_col="term_id",
    )
    assert isinstance(obs, pd.DataFrame)
    assert pd.testing.assert_frame_equal(obs, exp) is None
