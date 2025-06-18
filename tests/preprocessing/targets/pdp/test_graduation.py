import pandas as pd
import pytest

from student_success_tool.preprocessing.targets.pdp import graduation


@pytest.mark.parametrize(
    [
        "df",
        "intensity_time_limits",
        "num_terms_in_year",
        "max_term_rank",
        "student_id_cols",
        "exp",
    ],
    [
        # base case: all students labelable, one pos and one neg
        (
            pd.DataFrame(
                {
                    "student_id": ["01", "01", "01", "02", "02"],
                    "enrollment_intensity": ["PT", "FT", "FT", "PT", "PT"],
                    "years_to_degree": [4, 4, 4, 5, 5],
                    "enrollment_year": [1, 1, 1, 1, 1],
                    "term_rank": [1, 2, 3, 1, 2],
                    "term_is_pre_cohort": [True, False, False, False, False],
                },
            ).astype({"student_id": "string", "enrollment_intensity": "string"}),
            {"FT": [3, "year"], "PT": [6, "year"]},
            2,
            15,
            "student_id",
            pd.Series(
                data=[True, False],
                index=pd.Index(["01", "02"], dtype="string", name="student_id"),
                dtype="boolean",
                name="target",
            ),
        ),
        # lower max target term so part-time student isn't labelable
        (
            pd.DataFrame(
                {
                    "student_id": ["01", "01", "01", "02", "02"],
                    "enrollment_intensity": ["PT", "FT", "FT", "PT", "PT"],
                    "years_to_degree": [4, 4, 4, 5, 5],
                    "enrollment_year": [1, 1, 1, 1, 1],
                    "term_rank": [1, 2, 3, 1, 2],
                    "term_is_pre_cohort": [True, False, False, False, False],
                },
            ).astype({"student_id": "string", "enrollment_intensity": "string"}),
            {"FT": [3, "year"], "PT": [6, "year"]},
            2,
            12,
            "student_id",
            pd.Series(
                data=[True],
                index=pd.Index(["01"], dtype="string", name="student_id"),
                dtype="boolean",
                name="target",
            ),
        ),
        # one time limit for all enrollment intensities
        (
            pd.DataFrame(
                {
                    "student_id": ["01", "01", "01", "02", "02"],
                    "enrollment_intensity": ["PT", "FT", "FT", "PT", "PT"],
                    "years_to_degree": [4, 4, 4, 5, 5],
                    "enrollment_year": [1, 1, 1, 1, 1],
                    "term_rank": [1, 2, 3, 1, 2],
                    "term_is_pre_cohort": [True, False, False, False, False],
                },
            ).astype({"student_id": "string", "enrollment_intensity": "string"}),
            {"*": [3, "year"]},
            2,
            15,
            "student_id",
            pd.Series(
                data=[True, True],
                index=pd.Index(["01", "02"], dtype="string", name="student_id"),
                dtype="boolean",
                name="target",
            ),
        ),
        # reduce full-time / increase part-time students' years-to-degree
        (
            pd.DataFrame(
                {
                    "student_id": ["01", "01", "01", "02", "02"],
                    "enrollment_intensity": ["PT", "FT", "FT", "PT", "PT"],
                    "years_to_degree": [3, 3, 3, 7, 7],
                    "enrollment_year": [1, 1, 1, 1, 1],
                    "term_rank": [1, 2, 3, 1, 2],
                    "term_is_pre_cohort": [True, False, False, False, False],
                },
            ).astype({"student_id": "string", "enrollment_intensity": "string"}),
            {"FT": [3, "year"], "PT": [6, "year"]},
            2,
            15,
            "student_id",
            pd.Series(
                data=[False, True],
                index=pd.Index(["01", "02"], dtype="string", name="student_id"),
                dtype="boolean",
                name="target",
            ),
        ),
        # pathological case: years-to-degree varies across student terms
        (
            pd.DataFrame(
                {
                    "student_id": ["01", "01", "01", "02", "02"],
                    "enrollment_intensity": ["PT", "FT", "FT", "PT", "PT"],
                    "years_to_degree": [4, 5, 5, pd.NA, 5],
                    "enrollment_year": [1, 1, 1, 1, 1],
                    "term_rank": [1, 2, 3, 1, 2],
                    "term_is_pre_cohort": [True, False, False, False, False],
                },
            ).astype({"student_id": "string", "enrollment_intensity": "string"}),
            {"FT": [3, "year"], "PT": [6, "year"]},
            2,
            15,
            "student_id",
            pd.Series(
                data=[True, False],
                index=pd.Index(["01", "02"], dtype="string", name="student_id"),
                dtype="boolean",
                name="target",
            ),
        ),
    ],
)
def test_compute_target(
    df, intensity_time_limits, num_terms_in_year, max_term_rank, student_id_cols, exp
):
    obs = graduation.compute_target(
        df,
        intensity_time_limits=intensity_time_limits,
        num_terms_in_year=num_terms_in_year,
        max_term_rank=max_term_rank,
        student_id_cols=student_id_cols,
        enrollment_intensity_col="enrollment_intensity",
        years_to_degree_col="years_to_degree",
        enrollment_year_col="enrollment_year",
    )
    assert isinstance(obs, pd.Series)
    print("obs:", obs)
    print("exp:", exp)
    assert pd.testing.assert_series_equal(obs, exp) is None
