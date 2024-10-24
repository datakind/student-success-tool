import functools as ft
import typing as t
from collections.abc import Sequence

import pandas as pd


def extract_short_cip_code(ser: pd.Series) -> pd.Series:
    # NOTE: this simpler form works, but the values aren't nearly as clean
    # return ser.str.slice(stop=2).str.strip(".")
    return (
        ser.str.extract(r"^(?P<subject_area>\d[\d.])[\d.]+$", expand=False)
        .str.strip(".")
        .astype("string")
    )


def frac_credits_earned(
    df: pd.DataFrame,
    *,
    earned_col: str = "num_credits_earned",
    attempted_col: str = "num_credits_attempted",
) -> pd.Series:
    return df[earned_col].div(df[attempted_col])


def compute_values_equal(ser: pd.Series, to: t.Any | list[t.Any]) -> pd.Series:
    return ser.isin(to) if isinstance(to, list) else ser.eq(to)


def merge_many_dataframes(
    dfs: Sequence[pd.DataFrame],
    *,
    on: str | list[str],
    how: t.Literal["left", "right", "outer", "inner"] = "inner",
    sort: bool = False,
) -> pd.DataFrame:
    """
    Merge 2+ dataframes using the same set of merge parameters for each operation.

    See Also:
        - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.merge.html
    """
    return ft.reduce(
        lambda left, right: pd.merge(left, right, on=on, how=how, sort=sort), dfs
    )
