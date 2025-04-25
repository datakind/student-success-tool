import functools as ft
import logging
import typing as t
from collections.abc import Collection

import pandas as pd

from ....utils import types
from . import constants, shared

LOGGER = logging.getLogger(__name__)


def add_features(
    df: pd.DataFrame,
    *,
    first_term_of_year: types.TermType = constants.DEFAULT_FIRST_TERM_OF_YEAR,  # type: ignore
    core_terms: set[types.TermType] = constants.DEFAULT_CORE_TERMS,  # type: ignore
    peak_covid_terms: set[tuple[str, str]] = constants.DEFAULT_PEAK_COVID_TERMS,
    year_col: str = "academic_year",
    term_col: str = "academic_term",
) -> pd.DataFrame:
    """
    Compute term-level features from pdp course dataset,
    and add as columns to ``df`` .

    Args:
        df
        first_term_of_year
        core_terms: Set of terms that together comprise the "core" of the academic year,
            in contrast with additional, usually shorter terms that may take place
            between core terms. Default value is {"FALL", "SPRING"}, which typically
            corresponds to a semester system; for schools on a trimester calendary,
            {"FALL", "WINTER", "SPRING"} is probably what you want.
        peak_covid_terms: Set of (year, term) pairs considered by the institution as
            occurring during "peak" COVID; for example, ``("2020-21", "SPRING")`` .
    """
    LOGGER.info("adding term features ...")
    noncore_terms: set[types.TermType] = set(df[term_col].unique()) - set(core_terms)
    df_term = (
        _get_unique_sorted_terms_df(df, year_col=year_col, term_col=term_col)
        # only need to compute features on unique terms, rather than at course-level
        # merging back into `df` afterwards ensures all rows have correct values
        .assign(
            term_id=ft.partial(shared.year_term, year_col=year_col, term_col=term_col),
            term_start_dt=ft.partial(
                shared.year_term_dt,
                col="term_id",
                bound="start",
                first_term_of_year=first_term_of_year,
            ),
            term_rank=ft.partial(term_rank, year_col=year_col, term_col=term_col),
            term_rank_core=ft.partial(
                term_rank,
                year_col=year_col,
                term_col=term_col,
                terms_subset=core_terms,
            ),
            term_rank_noncore=ft.partial(
                term_rank,
                year_col=year_col,
                term_col=term_col,
                terms_subset=noncore_terms,
            ),
            term_in_peak_covid=ft.partial(
                term_in_peak_covid,
                year_col=year_col,
                term_col=term_col,
                peak_covid_terms=peak_covid_terms,
            ),
            # yes, this is silly, but it helps a tricky feature computation later on
            term_is_core=ft.partial(
                term_in_subset, terms_subset=core_terms, term_col=term_col
            ),
            term_is_noncore=ft.partial(
                term_in_subset, terms_subset=noncore_terms, term_col=term_col
            ),
        )
    )
    return pd.merge(df, df_term, on=[year_col, term_col], how="inner")


def term_rank(
    df: pd.DataFrame,
    *,
    year_col: str = "academic_year",
    term_col: str = "academic_term",
    terms_subset: t.Optional[Collection[str]] = None,
) -> pd.Series:
    df_terms = (
        _get_unique_sorted_terms_df(df, year_col=year_col, term_col=term_col)
        if terms_subset is None
        else _get_unique_sorted_terms_df(
            df.loc[df[term_col].isin(terms_subset), :],
            year_col=year_col,
            term_col=term_col,
        )
    )
    df_terms_ranked = df_terms.assign(
        term_rank=lambda df: pd.Series(list(range(len(df))))
    )
    # left-join back into df, so this works if df rows are at the course *or* term level
    term_id_cols = [year_col, term_col]
    return (
        pd.merge(df[term_id_cols], df_terms_ranked, on=term_id_cols, how="left")
        .loc[:, "term_rank"]
        .rename(None)
        .astype("Int8")
    )


def term_in_peak_covid(
    df: pd.DataFrame,
    *,
    year_col: str = "academic_year",
    term_col: str = "academic_term",
    peak_covid_terms: set[tuple[str, str]],
) -> pd.Series:
    return pd.Series(
        pd.MultiIndex.from_frame(df[[year_col, term_col]]).isin(peak_covid_terms)
    )


def term_in_subset(
    df: pd.DataFrame, terms_subset: set[types.TermType], term_col: str = "academic_term"
) -> pd.Series:
    return df[term_col].isin(terms_subset).astype("boolean")


def _get_unique_sorted_terms_df(
    df: pd.DataFrame,
    *,
    year_col: str = "academic_year",
    term_col: str = "academic_term",
) -> pd.DataFrame:
    return (
        df[[year_col, term_col]]
        # dedupe more convenient op than groupby([year_col, term_col])
        .drop_duplicates(ignore_index=True)
        # null year and/or term values aren't relevant/useful here
        .dropna(axis="index", how="any")
        # assumes year col is alphanumerically sortable, term col is categorically ordered
        .sort_values(by=[year_col, term_col], ignore_index=True)
    )
