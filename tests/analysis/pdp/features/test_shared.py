import pandas as pd
import pytest

from student_success_tool.analysis.pdp.features import shared

@pytest.mark.parametrize(
        ["ser", "to", "exp"],
        [
            (
                pd.Series(["0", "1", "2", "3", "4"]),
                "0",
                pd.Series([True, False, False, False, False])
            ),
            (
                pd.Series(["0", "1", "2", "3", "4"]),
                ["0","1","F","W"],
                pd.Series([True, True, False, False, False])
            )
        ]
)
def test_compute_values_equal(ser, to, exp):
    obs = shared.compute_values_equal(ser, to)
    assert isinstance(obs, pd.Series) and not obs.empty
    assert obs.dtype == 'bool'
    assert obs.equals(exp) or obs.compare(exp).empty
