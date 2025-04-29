import pandas as pd
import pytest

from student_success_tool.preprocessing.features.pdp import term


@pytest.mark.parametrize(
    ["df", "first_term_of_year", "core_terms", "peak_covid_terms", "exp"],
    [
        (
            pd.DataFrame(
                {
                    "student_guid": ["123", "123", "123", "123", "456", "456"],
                    "course_id": ["X101", "Y101", "X202", "Y202", "Z101", "Z202"],
                    "academic_year": [
                        "2020-21",
                        "2020-21",
                        "2020-21",
                        "2020-21",
                        "2019-20",
                        "2022-23",
                    ],
                    "academic_term": [
                        "FALL",
                        "FALL",
                        "WINTER",
                        "SPRING",
                        "FALL",
                        "SUMMER",
                    ],
                }
            ).astype(
                {
                    "academic_term": pd.CategoricalDtype(
                        ["FALL", "WINTER", "SPRING", "SUMMER"], ordered=True
                    )
                }
            ),
            "FALL",
            {"FALL", "SPRING"},
            {("2020-21", "SPRING"), ("2021-22", "FALL")},
            pd.DataFrame(
                {
                    "student_guid": ["123", "123", "123", "123", "456", "456"],
                    "course_id": ["X101", "Y101", "X202", "Y202", "Z101", "Z202"],
                    "academic_year": [
                        "2020-21",
                        "2020-21",
                        "2020-21",
                        "2020-21",
                        "2019-20",
                        "2022-23",
                    ],
                    "academic_term": [
                        "FALL",
                        "FALL",
                        "WINTER",
                        "SPRING",
                        "FALL",
                        "SUMMER",
                    ],
                    "term_id": [
                        "2020-21 FALL",
                        "2020-21 FALL",
                        "2020-21 WINTER",
                        "2020-21 SPRING",
                        "2019-20 FALL",
                        "2022-23 SUMMER",
                    ],
                    "term_start_dt": pd.to_datetime(
                        [
                            "2020-09-01",
                            "2020-09-01",
                            "2021-01-01",
                            "2021-02-01",
                            "2019-09-01",
                            "2023-06-01",
                        ],
                    ),
                    "term_rank": [1, 1, 2, 3, 0, 4],
                    "term_rank_core": [1, 1, pd.NA, 2, 0, pd.NA],
                    "term_rank_noncore": [pd.NA, pd.NA, 0, pd.NA, pd.NA, 1],
                    "term_in_peak_covid": [False, False, False, True, False, False],
                    "term_is_core": [True, True, False, True, True, False],
                    "term_is_noncore": [False, False, True, False, False, True],
                }
            ).astype(
                {
                    "term_rank_core": "Int8",
                    "term_rank_noncore": "Int8",
                    "academic_term": pd.CategoricalDtype(
                        ["FALL", "WINTER", "SPRING", "SUMMER"], ordered=True
                    ),
                }
            ),
        ),
    ],
)
def test_add_term_features(df, first_term_of_year, core_terms, peak_covid_terms, exp):
    obs = term.add_features(
        df,
        first_term_of_year=first_term_of_year,
        core_terms=core_terms,
        peak_covid_terms=peak_covid_terms,
    )
    assert isinstance(obs, pd.DataFrame)
    assert pd.testing.assert_frame_equal(obs, exp, check_dtype=False) is None


@pytest.mark.parametrize(
    ["df", "year_col", "term_col", "terms_subset", "exp"],
    [
        (
            pd.DataFrame(
                {
                    "year": ["22-23", "23-24", "22-23", "21-22", "22-23", None],
                    "term": ["FA", "FA", "SP", "SP", "FA", None],
                }
            ).astype({"term": pd.CategoricalDtype(["FA", "SP"], ordered=True)}),
            "year",
            "term",
            None,
            pd.Series([1, 3, 2, 0, 1, pd.NA], dtype="Int8"),
        ),
    ],
)
def test_term_rank(df, year_col, term_col, terms_subset, exp):
    obs = term.term_rank(
        df, year_col=year_col, term_col=term_col, terms_subset=terms_subset
    )
    assert isinstance(obs, pd.Series) and not obs.empty
    assert obs.equals(exp) or obs.compare(exp).empty


@pytest.mark.parametrize(
    ["df", "year_col", "term_col", "peak_covid_terms", "exp"],
    [
        (
            pd.DataFrame(
                {
                    "year": ["20-21", "20-21", "21-22", "21-22", "21-22"],
                    "term": ["FA", "SP", "FA", "SP", "SU"],
                }
            ),
            "year",
            "term",
            [("20-21", "SP"), ("20-21", "SU"), ("21-22", "FA")],
            pd.Series([False, True, True, False, False], dtype="bool"),
        )
    ],
)
def test_term_in_peak_covid(df, year_col, term_col, peak_covid_terms, exp):
    obs = term.term_in_peak_covid(
        df, year_col=year_col, term_col=term_col, peak_covid_terms=peak_covid_terms
    )
    assert isinstance(obs, pd.Series) and not obs.empty
    assert obs.equals(exp) or obs.compare(exp).empty


@pytest.mark.parametrize(
    ["df", "terms_subset", "term_col", "exp"],
    [
        (
            pd.DataFrame({"term": ["FALL", "WINTER", "SPRING", "SUMMER"]}),
            {"FALL", "SPRING"},
            "term",
            pd.Series([True, False, True, False], dtype="boolean"),
        ),
        (
            pd.DataFrame({"term": ["FALL", "WINTER", "SPRING", "SUMMER"]}),
            {"FALL", "WINTER", "SPRING"},
            "term",
            pd.Series([True, True, True, False], dtype="boolean"),
        ),
    ],
)
def test_term_in_subset(df, terms_subset, term_col, exp):
    obs = term.term_in_subset(df, terms_subset=terms_subset, term_col=term_col)
    assert isinstance(obs, pd.Series)
    assert pd.testing.assert_series_equal(obs, exp, check_names=False) is None
