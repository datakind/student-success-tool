import pandas as pd
import pytest

from student_success_tool.preprocessing.features.pdp import shared


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


@pytest.mark.parametrize(
    ["df", "year_col", "term_col", "exp"],
    [
        (
            pd.DataFrame(
                {
                    "year": ["20-21", "20-21", "21-22", "21-22"],
                    "term": ["FA", "WI", "SP", "SU"],
                }
            ),
            "year",
            "term",
            pd.Series(["20-21 FA", "20-21 WI", "21-22 SP", "21-22 SU"]),
        )
    ],
)
def test_year_term(df, year_col, term_col, exp):
    obs = shared.year_term(df, year_col=year_col, term_col=term_col)
    assert isinstance(obs, pd.Series) and not obs.empty
    assert obs.equals(exp) or obs.compare(exp).empty


@pytest.mark.parametrize(
    ["df", "col", "bound", "first_term_of_year", "exp"],
    [
        (
            pd.DataFrame(
                {
                    "yt": [
                        "2020-21 FALL",
                        "2020-21 WINTER",
                        "2020-21 SPRING",
                        "2020-21 SUMMER",
                    ]
                }
            ),
            "yt",
            "start",
            "FALL",
            pd.Series(
                ["2020-09-01", "2021-01-01", "2021-02-01", "2021-06-01"],
                dtype="datetime64[ns]",
            ),
        ),
        (
            pd.DataFrame(
                {
                    "yt": [
                        "2021-22 FALL",
                        "2021-22 WINTER",
                        "2021-22 SPRING",
                        "2021-22 SUMMER",
                    ]
                }
            ),
            "yt",
            "end",
            "SUMMER",
            pd.Series(
                ["2021-12-31", "2022-01-31", "2022-05-31", "2021-08-31"],
                dtype="datetime64[ns]",
            ),
        ),
    ],
)
def test_year_term_dt(df, col, bound, first_term_of_year, exp):
    obs = shared.year_term_dt(
        df, col=col, bound=bound, first_term_of_year=first_term_of_year
    )
    assert isinstance(obs, pd.Series) and not obs.empty
    assert obs.equals(exp) or obs.compare(exp).empty
