import functools as ft

import pandas as pd
import pytest

from student_success_tool.preprocessing import checkpoints
from student_success_tool.preprocessing.targets.pdp import credits_earned


@pytest.mark.parametrize(
    [
        "df",
        "min_num_credits",
        "checkpoint",
        "intensity_time_limits",
        "num_terms_in_year",
        "max_term_rank",
        "exp",
    ],
    [
        # base case
        (
            pd.DataFrame(
                {
                    "student_id": ["01", "01", "01", "01"],
                    "enrollment_intensity": ["FT", "FT", "FT", "FT"],
                    "num_credits": [12, 24, 36, 48],
                    "term_rank": [1, 2, 3, 5],
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
            "infer",
            pd.Series(
                data=[False],
                index=pd.Index(["01"], dtype="string", name="student_id"),
                name="target",
                dtype="boolean",
            ),
        ),
        # multiple students, one full-time the other part-time
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
            "infer",
            pd.Series(
                data=[True, True],
                index=pd.Index(["01", "02"], dtype="string", name="student_id"),
                name="target",
                dtype="boolean",
            ),
        ),
        # checkpoint given as callable
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
                checkpoints.pdp.first_student_terms,
                student_id_cols="student_id",
                sort_cols="term_rank",
                include_cols=["enrollment_intensity", "num_credits"],
            ),
            {"FT": [4, "term"], "PT": [8, "term"]},
            2,
            "infer",
            pd.Series(
                data=[True, True],
                index=pd.Index(["01", "02"], dtype="string", name="student_id"),
                name="target",
                dtype="boolean",
            ),
        ),
        # not enough terms in dataset to compute target
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
            "infer",
            pd.Series(
                data=[],
                index=pd.Index([], dtype="string", name="student_id"),
                name="target",
                dtype="boolean",
            ),
        ),
        # one time limit for all enrollment intensities
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
            10,
            pd.Series(
                data=[True, False],
                index=pd.Index(["01", "02"], dtype="string", name="student_id"),
                name="target",
                dtype="boolean",
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
    max_term_rank,
    exp,
):
    obs = credits_earned.compute_target(
        df,
        min_num_credits=min_num_credits,
        checkpoint=checkpoint,
        intensity_time_limits=intensity_time_limits,
        num_terms_in_year=num_terms_in_year,
        max_term_rank=max_term_rank,
        student_id_cols="student_id",
        enrollment_intensity_col="enrollment_intensity",
        num_credits_col="num_credits",
        term_rank_col="term_rank",
    )
    assert isinstance(obs, pd.Series)
    assert pd.testing.assert_series_equal(obs, exp) is None
