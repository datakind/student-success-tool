import functools as ft
import logging
import typing as t

import pandas as pd

from .. import constants
from . import shared

LOGGER = logging.getLogger(__name__)


def add_features(
    df: pd.DataFrame,
    *,
    year_col: str = "academic_year",
    term_col: str = "academic_term",
    peak_covid_terms: set[tuple[str, str]] = constants.DEFAULT_PEAK_COVID_TERMS,
) -> pd.DataFrame:
    """
    Compute term-level features from pdp course dataset,
    and add as columns to ``df`` .

    Args:
        df
        year_col: Name of column in ``df`` containing the records' academic year.
        term_col: Name of column in ``df`` containing the records' academic term.
        peak_covid_terms: Set of (year, term) pairs considered by the institution as
            occurring during "peak" COVID; for example, ``("2020-21", "SPRING")`` .
    """
    LOGGER.info("adding term features ...")
    df_term = (
        _get_unique_sorted_terms_df(df, year_col=year_col, term_col=term_col)
        # only need to compute features on unique terms, rather than at course-level
        # merging back into `df` afterwards ensures all rows have correct values
        .assign(
            term_id=ft.partial(term_id, year_col=year_col, term_col=term_col),
            term_rank=ft.partial(term_rank, year_col=year_col, term_col=term_col),
            term_rank_fall_spring=ft.partial(
                term_rank,
                year_col=year_col,
                term_col=term_col,
                terms_subset=["FALL", "SPRING"],
            ),
            term_in_peak_covid=ft.partial(
                term_in_peak_covid,
                year_col=year_col,
                term_col=term_col,
                peak_covid_terms=peak_covid_terms,
            ),
            # yes, this is silly, but it helps a tricky feature computation later on
            term_is_fall_spring=ft.partial(term_is_fall_spring, term_col=term_col),
        )
    )
    df_term_agg = (
        df.groupby(by=[year_col, term_col], as_index=False, observed=True)
        .agg(
            term_course_begin_date_min=term_course_begin_date_min_col_agg(),
            term_course_end_date_max=term_course_end_date_max_col_agg(),
        )
    )  # fmt: skip
    return shared.merge_many_dataframes(
        [df, df_term, df_term_agg], on=[year_col, term_col], how="inner"
    )


def term_id(
    df: pd.DataFrame,
    *,
    year_col: str = "academic_year",
    term_col: str = "academic_term",
) -> pd.Series:
    return df[year_col].str.cat(df[term_col], sep=" ")


def term_rank(
    df: pd.DataFrame,
    *,
    year_col: str = "academic_year",
    term_col: str = "academic_term",
    terms_subset: t.Optional[list[str]] = None,
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


def term_is_fall_spring(
    df: pd.DataFrame, *, term_col: str = "academic_term"
) -> pd.Series:
    return df[term_col].isin(["FALL", "SPRING"])


def term_course_begin_date_min_col_agg(col: str = "course_begin_date") -> pd.NamedAgg:
    return pd.NamedAgg(col, "min")


def term_course_end_date_max_col_agg(col: str = "course_end_date") -> pd.NamedAgg:
    return pd.NamedAgg(col, "max")


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
