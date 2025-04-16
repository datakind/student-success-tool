import functools as ft

import pandas as pd
import pytest

from student_success_tool.targets.pdp import credits_earned, shared


@pytest.mark.parametrize(
    [
        "df",
        "min_num_credits",
        "checkpoint",
        "intensity_time_limits",
        "num_terms_in_year",
        "student_id_cols",
        "exp",
    ],
    [
        (
            pd.DataFrame(
                {
                    "student_id": ["01", "01", "01", "01"],
                    "enrollment_intensity": ["FT", "FT", "FT", "FT"],
                    "num_credits": [12, 24, 36, 48],
                    "term_rank": [1, 2, 3, 4],
                },
            ).astype({"student_id": "string", "enrollment_intensity": "string"}),
            45.0,
            # first term as checkpoint
            pd.DataFrame(
                {
                    "student_id": ["01"],
                    "enrollment_intensity": ["FT"],
                    "num_credits": [12],
                    "term_rank": [1],
                },
            ).astype({"student_id": "string", "enrollment_intensity": "string"}),
            {"FT": [4, "term"]},
            2,
            "student_id",
            pd.Series(
                data=[False],
                index=pd.Index(["01"], dtype="string", name="student_id"),
                name="target",
            ),
        ),
        (
            pd.DataFrame(
                {
                    "student_id": ["01", "01", "02", "02"],
                    "enrollment_intensity": ["FT", "FT", "PT", "PT"],
                    "num_credits": [12, 48, 8, 64],
                    "term_rank": [1, 4, 1, 10],
                },
            ).astype({"student_id": "string", "enrollment_intensity": "string"}),
            60.0,
            # first term as checkpoint
            pd.DataFrame(
                {
                    "student_id": ["01", "02"],
                    "enrollment_intensity": ["FT", "PT"],
                    "num_credits": [12, 8],
                    "term_rank": [1, 1],
                },
            ).astype({"student_id": "string", "enrollment_intensity": "string"}),
            {"FT": [4, "term"], "PT": [8, "term"]},
            2,
            "student_id",
            pd.Series(
                data=[True, True],
                index=pd.Index(["01", "02"], dtype="string", name="student_id"),
                name="target",
            ),
        ),
        (
            pd.DataFrame(
                {
                    "student_id": ["01", "01", "02", "02"],
                    "enrollment_intensity": ["FT", "FT", "PT", "PT"],
                    "num_credits": [12, 48, 8, 64],
                    "term_rank": [1, 4, 1, 10],
                },
            ).astype({"student_id": "string", "enrollment_intensity": "string"}),
            60.0,
            # first term as checkpoint via callable
            ft.partial(
                shared.get_first_student_terms,
                student_id_cols="student_id",
                sort_cols="term_rank",
                include_cols=["enrollment_intensity", "num_credits"],
            ),
            {"FT": [4, "term"], "PT": [8, "term"]},
            2,
            "student_id",
            pd.Series(
                data=[True, True],
                index=pd.Index(["01", "02"], dtype="string", name="student_id"),
                name="target",
            ),
        ),
        (
            pd.DataFrame(
                {
                    "student_id": ["01", "01", "01", "01"],
                    "enrollment_intensity": ["FT", "PT", "PT", "FT"],
                    "num_credits": [12, 24, 36, 48],
                    "term_rank": [1, 2, 6, 7],
                },
            ).astype({"student_id": "string", "enrollment_intensity": "string"}),
            45.0,
            # second term as checkpoint
            pd.DataFrame(
                {
                    "student_id": ["01"],
                    "enrollment_intensity": ["PT"],
                    "num_credits": [24],
                    "term_rank": [3],
                },
            ).astype({"student_id": "string", "enrollment_intensity": "string"}),
            {"FT": [4, "term"], "PT": [8, "term"]},
            4,
            "student_id",
            pd.Series(
                data=[False],
                index=pd.Index(["01"], dtype="string", name="student_id"),
                name="target",
            ),
        ),
        (
            pd.DataFrame(
                {
                    "student_id": ["01", "01", "02", "02"],
                    "enrollment_intensity": ["FT", "FT", "PT", "PT"],
                    "num_credits": [12, 48, 8, 64],
                    "term_rank": [1, 4, 1, 8],
                },
            ).astype({"student_id": "string", "enrollment_intensity": "string"}),
            60.0,
            # first term as checkpoint
            pd.DataFrame(
                {
                    "student_id": ["01", "02"],
                    "enrollment_intensity": ["FT", "PT"],
                    "num_credits": [12, 8],
                    "term_rank": [1, 1],
                },
            ).astype({"student_id": "string", "enrollment_intensity": "string"}),
            {"*": [8, "term"]},
            2,
            "student_id",
            pd.Series(
                data=[True, False],
                index=pd.Index(["01", "02"], dtype="string", name="student_id"),
                name="target",
            ),
        ),
    ],
)
def test_compute_target(
    df,
    min_num_credits,
    checkpoint,
    intensity_time_limits,
    num_terms_in_year,
    student_id_cols,
    exp,
):
    obs = credits_earned.compute_target(
        df,
        min_num_credits=min_num_credits,
        checkpoint=checkpoint,
        intensity_time_limits=intensity_time_limits,
        num_terms_in_year=num_terms_in_year,
        student_id_cols=student_id_cols,
        enrollment_intensity_col="enrollment_intensity",
        num_credits_col="num_credits",
        term_rank_col="term_rank",
    )
    assert isinstance(obs, pd.Series)
    assert pd.testing.assert_series_equal(obs, exp) is None
