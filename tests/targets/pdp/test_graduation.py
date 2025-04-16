import pandas as pd
import pytest

from student_success_tool.targets.pdp import graduation


@pytest.mark.parametrize(
    ["df", "intensity_time_limits", "num_terms_in_year", "student_id_cols", "exp"],
    [
        (
            pd.DataFrame(
                {
                    "student_id": ["01", "01", "01", "03", "02", "02"],
                    "enrollment_intensity": ["FT", "PT", "FT", "FT", "PT", "FT"],
                    "years_to_degree": [4, 4, 4, 5, 6, 6],
                    "enrollment_year": [1, 1, 1, 1, 1, 1],
                },
            ).astype({"student_id": "string", "enrollment_intensity": "string"}),
            {"FT": [4, "year"], "PT": [12, "term"]},
            2,
            "student_id",
            pd.Series(
                data=[False, True, True],
                index=pd.Index(["01", "03", "02"], dtype="string", name="student_id"),
                name="target",
            ),
        ),
        (
            pd.DataFrame(
                {
                    "student_id": ["01", "02", "03"],
                    "enrollment_intensity": ["FT", "PT", "FT"],
                    "years_to_degree": [2, 4, 3],
                    "enrollment_year": [1, 1, 1],
                },
            ).astype({"student_id": "string", "enrollment_intensity": "string"}),
            {"FT": [8, "term"], "PT": [16, "term"]},
            4,
            "student_id",
            pd.Series(
                data=[False, False, True],
                index=pd.Index(["01", "02", "03"], dtype="string", name="student_id"),
                name="target",
            ),
        ),
        (
            pd.DataFrame(
                {
                    "student_id": ["01", "02"],
                    "enrollment_intensity": ["FT", "FT"],
                    "years_to_degree": [1, pd.NA],
                    "enrollment_year": [1, 1],
                },
            ).astype({"student_id": "string", "enrollment_intensity": "string"}),
            {"FT": [2, "year"]},
            4,
            "student_id",
            pd.Series(
                data=[False, True],
                index=pd.Index(["01", "02"], dtype="string", name="student_id"),
                name="target",
            ),
        ),
        (
            pd.DataFrame(
                {
                    "student_id1": ["01", "02", "03"],
                    "student_id2": ["A", "B", "A"],
                    "enrollment_intensity": ["FT", "PT", "FT"],
                    "years_to_degree": [2, 4, 3],
                    "enrollment_year": [1, 1, 1],
                },
            ).astype(
                {
                    "student_id1": "string",
                    "student_id2": "string",
                    "enrollment_intensity": "string",
                }
            ),
            {"FT": [2, "year"], "PT": [4, "year"]},
            3,
            ["student_id1", "student_id2"],
            pd.Series(
                data=[False, False, True],
                index=pd.MultiIndex.from_frame(
                    pd.DataFrame(
                        {
                            "student_id1": ["01", "02", "03"],
                            "student_id2": ["A", "B", "A"],
                        },
                        dtype="string",
                    )
                ),
                name="target",
            ),
        ),
        (
            pd.DataFrame(
                {
                    "student_id": ["01", "02"],
                    "enrollment_intensity": ["FT", "PT"],
                    "years_to_degree": [1, 3],
                    "enrollment_year": [1, 1],
                },
            ).astype({"student_id": "string", "enrollment_intensity": "string"}),
            {"*": [2, "year"]},
            4,
            "student_id",
            pd.Series(
                data=[False, True],
                index=pd.Index(["01", "02"], dtype="string", name="student_id"),
                name="target",
            ),
        ),
        # this is a pathological case: years-to-degree varies across student-terms
        (
            pd.DataFrame(
                {
                    "student_id": ["01", "01", "01"],
                    "enrollment_intensity": ["FT", "PT", "FT"],
                    "years_to_degree": [4, 5, pd.NA],
                    "enrollment_year": [1, 1, 1],
                },
            ).astype(
                {
                    "student_id": "string",
                    "enrollment_intensity": "string",
                    "years_to_degree": "Int8",
                }
            ),
            {"FT": [4, "year"]},
            4,
            "student_id",
            pd.Series(
                data=[False],
                index=pd.Index(["01"], dtype="string", name="student_id"),
                name="target",
            ),
        ),
    ],
)
def test_compute_target(
    df, intensity_time_limits, num_terms_in_year, student_id_cols, exp
):
    obs = graduation.compute_target(
        df,
        intensity_time_limits=intensity_time_limits,
        num_terms_in_year=num_terms_in_year,
        student_id_cols=student_id_cols,
        enrollment_intensity_col="enrollment_intensity",
        years_to_degree_col="years_to_degree",
        enrollment_year_col="enrollment_year",
    )
    assert isinstance(obs, pd.Series)
    assert pd.testing.assert_series_equal(obs, exp) is None
