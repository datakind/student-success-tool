import functools as ft
import re
import typing as t
from collections.abc import Sequence
from datetime import date

import pandas as pd

from ....utils import types

RE_YEAR_TERM = re.compile(
    r"(?P<start_yr>\d{4})-(?P<end_yr>\d{2}) (?P<term>FALL|WINTER|SPRING|SUMMER)",
    flags=re.IGNORECASE,
)

TERM_BOUND_MONTH_DAYS = {
    "FALL": {"start": (9, 1), "end": (12, 31)},
    "WINTER": {"start": (1, 1), "end": (1, 31)},
    "SPRING": {"start": (2, 1), "end": (5, 31)},
    "SUMMER": {"start": (6, 1), "end": (8, 31)},
}


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


def year_term(df: pd.DataFrame, *, year_col: str, term_col: str) -> pd.Series:
    return df[year_col].str.cat(df[term_col], sep=" ")


def year_term_dt(
    df: pd.DataFrame,
    *,
    col: str,
    bound: t.Literal["start", "end"],
    first_term_of_year: types.TermType,
) -> pd.Series:
    """
    Compute an approximate start/end date for a given year-term,
    e.g. to compute time elapsed between course enrollments or to order course history.

    Args:
        df
        col: Column in ``df`` whose values represent academic/cohort year and term,
            formatted as "YYYY-YY TERM".
        bound: Which bound of the date range spanned by ``year_term`` to return;
            either the start (left) or end (right) bound.
        first_term_of_year: Term that officially begins the institution's academic year,
            either "FALL" or "SUMMER", which determines how the date's year is assigned.

    See Also:
        - :func:`year_term()`
    """
    return (
        df[col]
        .map(
            ft.partial(
                _year_term_to_dt, bound=bound, first_term_of_year=first_term_of_year
            ),
            na_action="ignore",
        )
        .astype("datetime64[s]")
    )


def _year_term_to_dt(
    year_term: str, bound: t.Literal["start", "end"], first_term_of_year: types.TermType
) -> date:
    if match := RE_YEAR_TERM.search(year_term):
        start_yr = int(match["start_yr"])
        term = match["term"].upper()
        yr = (
            start_yr
            if (term == "FALL" or (term == "SUMMER" and first_term_of_year == "SUMMER"))
            else start_yr + 1
        )
        mo, dy = TERM_BOUND_MONTH_DAYS[term][bound]
        return date(yr, mo, dy)
    else:
        raise ValueError(f"invalid year_term value: {year_term}")
