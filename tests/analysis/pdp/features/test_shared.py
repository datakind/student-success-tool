import pandas as pd
import pytest

from student_success_tool.analysis.pdp.features import shared


@pytest.mark.parametrize(
    ["ser", "to", "exp"],
    [
        (
            pd.Series(["0", "1", "2", "3", "4"]),
            "0",
            pd.Series([True, False, False, False, False]),
        ),
        (
            pd.Series(["0", "1", "2", "3", "4"]),
            ["0", "1", "F", "W"],
            pd.Series([True, True, False, False, False]),
        ),
    ],
)
def test_compute_values_equal(ser, to, exp):
    obs = shared.compute_values_equal(ser, to)
    assert isinstance(obs, pd.Series) and not obs.empty
    assert obs.dtype == "bool"
    assert obs.equals(exp) or obs.compare(exp).empty


@pytest.mark.parametrize(
    ["df", "earned_col", "attempted_col", "exp"],
    [
        (
            pd.DataFrame(
                {
                    "nc_earned": [5.0, 7.5, 6.0, 0.0],
                    "nc_attempted": [10.0, 7.5, 8.0, 15.0],
                }
            ),
            "nc_earned",
            "nc_attempted",
            pd.Series([0.5, 1.0, 0.75, 0.0]),
        ),
    ],
)
def test_frac_credits_earned(df, earned_col, attempted_col, exp):
    obs = shared.frac_credits_earned(
        df, earned_col=earned_col, attempted_col=attempted_col
    )
    assert isinstance(obs, pd.Series) and not obs.empty
    assert obs.equals(exp) or obs.compare(exp).empty
