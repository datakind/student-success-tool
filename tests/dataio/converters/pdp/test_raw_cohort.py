import pandas as pd
import pytest

from student_success_tool.dataio.converters import pdp


@pytest.fixture(scope="module")
def df_test():
    return pd.DataFrame(
        {
            "student_id": ["01", "02"],
            "cohort": ["2020-21", "2024-25"],
            "attemptedgatewaymathyear_1": ["Y", "N"],
            "gatewayenglishgradey_1": ["P", "F"],
            "someothermangledcolumn": ["X", "Y"],
        },
        dtype="string",
    )


@pytest.mark.parametrize(
    ["overrides", "exp"],
    [
        (
            None,
            pd.DataFrame(
                {
                    "student_id": ["01", "02"],
                    "cohort": ["2020-21", "2024-25"],
                    "attempted_gateway_math_year_1": ["Y", "N"],
                    "gateway_english_grade_y_1": ["P", "F"],
                    "someothermangledcolumn": ["X", "Y"],
                },
                dtype="string",
            ),
        ),
        (
            {
                "someothermangledcolumn": "some_other_mangled_column",
                "missingcol": "missing_col",
            },
            pd.DataFrame(
                {
                    "student_id": ["01", "02"],
                    "cohort": ["2020-21", "2024-25"],
                    "attempted_gateway_math_year_1": ["Y", "N"],
                    "gateway_english_grade_y_1": ["P", "F"],
                    "some_other_mangled_column": ["X", "Y"],
                },
                dtype="string",
            ),
        ),
    ],
)
def test_rename_mangled_column_names(df_test, overrides, exp):
    obs = pdp.rename_mangled_column_names(df_test, overrides=overrides)
    assert isinstance(obs, pd.DataFrame)
    assert pd.testing.assert_frame_equal(obs, exp) is None
