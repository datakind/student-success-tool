import functools as ft
import itertools
import logging
from collections.abc import Iterable
import typing as t

import pandas as pd
from pandas.core.groupby import DataFrameGroupBy

from .... import utils
from . import constants

LOGGER = logging.getLogger(__name__)


def add_features(
    df: pd.DataFrame, *, student_id_cols: list[str], sort_cols: list[str]
) -> pd.DataFrame:
    LOGGER.info("adding student-term cumulative features ...")
    # sort so that student-terms are ordered chronologically
    df = df.sort_values(by=student_id_cols + sort_cols, ignore_index=True)
    # specifically *don't* re-sort when grouping
    df_grped = df.groupby(by=student_id_cols, as_index=True, observed=True, sort=False)
    num_course_cols = [
        col
        for col in df.columns
        if col.startswith(f"{constants.NUM_COURSE_FEATURE_COL_PREFIX}_")
    ]
    dummy_course_cols = [
        col
        for col in df.columns
        if col.startswith(f"{constants.DUMMY_COURSE_FEATURE_COL_PREFIX}_")
    ]
    df_expanding_agg = (
        expanding_agg_features(
            df_grped,
            num_course_cols=num_course_cols,
            dummy_course_cols=dummy_course_cols,
            col_aggs=[
                ("term_id", "count"),
                ("term_in_peak_covid", "sum"),
                ("term_is_core", "sum"),
                ("term_is_noncore", "sum"),
                ("term_is_while_student_enrolled_at_other_inst", "sum"),
                ("term_is_pre_cohort", "sum"),
                ("course_level_mean", ["mean", "min", "std"]),
                ("course_grade_numeric_mean", ["mean", "min", "std"]),
                ("num_courses", ["sum", "mean", "min"]),
                ("num_credits_attempted", ["sum", "mean", "min"]),
                ("num_credits_earned", ["sum", "mean", "min"]),
                ("student_pass_rate_above_sections_avg", "sum"),
                ("student_completion_rate_above_sections_avg", "sum"),
            ],
            credits=constants.DEFAULT_COURSE_CREDIT_CHECK,
        )
        # rename/dtype special cols for clarity in downstream calcs
        .astype(
            {
                "cumcount_term_id": "Int8",
                "cumsum_term_is_core": "Int8",
                "cumsum_term_is_noncore": "Int8",
            }
        )
        .rename(
            columns={
                "cumcount_term_id": "cumnum_terms_enrolled",
                "cumsum_term_is_core": "cumnum_core_terms_enrolled",
                "cumsum_term_is_noncore": "cumnum_noncore_terms_enrolled",
            }
        )
    )
    df_cumnum_ur = cumnum_unique_and_repeated_features(
        df_grped, cols=["course_ids", "course_subjects", "course_subject_areas"]
    )
    concat_dfs = [df, df_cumnum_ur, df_expanding_agg]
    return (
        # despite best efforts, the student-id index is dropped from df_cumnum_ur
        # and, through sheer pandas insanity, merge on student_id_cols produces
        # huge numbers of duplicate rows -- truly impossible shit, that
        # however, by definition, these transforms shouldn't alter the original indexing, so:
        pd.concat(concat_dfs, axis="columns")
        # add a last couple features, which don't fit nicely into above logic
        .pipe(add_cumfrac_terms_enrolled_features, student_id_cols=student_id_cols)
        .pipe(
            add_term_diff_features,
            cols=[
                "num_courses",
                "num_credits_earned",
                "course_grade_numeric_mean",
                "course_level_mean",
            ],
            max_term_num=4,
            student_id_cols=student_id_cols,
        )
    )


def expanding_agg_features(
    df_grped: DataFrameGroupBy,
    *,
    num_course_cols: list[str],
    col_aggs: list[tuple[str, str | list[str]]],
    credits: t.Optional[int] = constants.DEFAULT_COURSE_CREDIT_CHECK,
    dummy_course_cols: t.Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Compute various aggregate features over an expanding window per (student) group.

    Args:
        df_grped
        num_course_cols
        col_aggs
        credits: the number of credits to check if courses of interest were taken within
        dummy_course_cols: the columns that were checked for whether a course was taken to check if they were taken within the number of credits of interest
    """
    LOGGER.info("computing expanding window aggregate features ...")
    agg_dict = dict(col_aggs) | {col: "sum" for col in num_course_cols}
    if dummy_course_cols is not None:
        agg_dict |= {col: "max" for col in dummy_course_cols}

    df_cumaggs = (
        df_grped.expanding()
        .agg(agg_dict)
        # pandas does weird stuff when indexing on windowed operations
        # this should get us back to student_id_cols only on the index
        .reset_index(level=-1, drop=True)
    )

    # unfortunately, expanding doesn't support the "named agg" syntax
    # so we have to (flatten and) rename the columns manually
    # df_cumaggs.columns = [f"{col}_cum{fn}" for col, fn in df_cumaggs.columns]
    df_cumaggs.columns = [f"cum{fn}_{col}" for col, fn in df_cumaggs.columns]
    # for all num_courses_*_cumsum columns, divide by num_courses_cumsum
    # to get equivalent cumfracs, whose relative values are easier for models to learn from
    num_courses_cumsum_cols = [
        col
        for col in df_cumaggs.columns
        if (
            col.startswith(f"cumsum_{constants.NUM_COURSE_FEATURE_COL_PREFIX}_")
            and col != "cumsum_num_courses"  # HACK
        )
    ]
    df_cumfracs = (
        df_cumaggs[num_courses_cumsum_cols]
        .div(df_cumaggs["cumsum_num_courses"], axis="index")
        .rename(columns=lambda col: col.replace("cumsum", "cumfrac"))
    )
    concat_dfs = [df_cumaggs, df_cumfracs]
    if dummy_course_cols is not None:
        action_cols = [f"cummax_{dummy_course}" for dummy_course in dummy_course_cols]
        action_status_df = pd.DataFrame(index=df_cumaggs.index)

        for col in action_cols:
            # within_col = f"{col}_in_{credits}_creds"
            within_col = col.replace("cummax_", f"cummax_in_{credits}_creds_")
            action_status_df[within_col] = (df_cumaggs[col].astype(bool)) & (
                df_cumaggs["cumsum_num_credits_earned"] <= credits
            )

        action_status_df = action_status_df.groupby(
            level=df_cumaggs.index.names
        ).transform("max")
        concat_dfs.append(action_status_df)

    return (
        # append cum-frac features to all cum-agg features
        pd.concat(concat_dfs, axis="columns")
        # *drop* the original cumsum columns, since the derived cumfracs are sufficient
        .drop(columns=num_courses_cumsum_cols)
        # drop our student-id(s) index, it just causes trouble
        .reset_index(drop=True)
    )


def cumnum_unique_and_repeated_features(
    df_grped: DataFrameGroupBy, *, cols: list[str]
) -> pd.DataFrame:
    """
    Compute the cumulative number of repeated elements within each (student) group
    for a set of columns whose values are per-row lists of elements.

    Args:
        df_grped
        cols
    """
    LOGGER.info("computing cumulative elements features ...")
    df_accumulated_lists = df_grped[cols].transform(_expand_elements)
    return df_accumulated_lists.assign(
        **{
            f"cumnum_unique_{col}": ft.partial(num_unique_elements, col=col)
            for col in cols
        }
        | {
            f"cumnum_repeated_{col}": ft.partial(num_repeated_elements, col=col)
            for col in cols
        }
    ).drop(columns=cols)


def num_unique_elements(df: pd.DataFrame, *, col: str) -> pd.Series:
    return df[col].map(lambda eles: len(set(eles)), na_action="ignore").astype("Int16")


def num_repeated_elements(df: pd.DataFrame, *, col: str) -> pd.Series:
    return (
        df[col]
        .map(lambda eles: len(eles) - len(set(eles)), na_action="ignore")
        .astype("Int16")
    )


def _expand_elements(x: list) -> list:
    return list(itertools.accumulate(x, func=_concat_elements))


def _concat_elements(*args: Iterable) -> list:
    return list(itertools.chain(*args))


def add_cumfrac_terms_enrolled_features(
    df: pd.DataFrame, *, student_id_cols: list[str]
) -> pd.DataFrame:
    LOGGER.info("computing cumfrac terms enrolled features ...")
    df_grped = df.groupby(by=student_id_cols, as_index=False, sort=False)
    df_min_term_ranks = df_grped.agg(
        min_student_term_rank=("term_rank", "min"),
        min_student_term_rank_core=("term_rank_core", "min"),
        min_student_term_rank_noncore=("term_rank_noncore", "min"),
    )
    return pd.merge(df, df_min_term_ranks, on=student_id_cols, how="inner").assign(
        cumfrac_terms_enrolled=ft.partial(
            _compute_cumfrac_terms_enrolled,
            term_rank_col="term_rank",
            min_student_term_rank_col="min_student_term_rank",
            cumnum_terms_enrolled_col="cumnum_terms_enrolled",
        ),
        cumfrac_core_terms_enrolled=ft.partial(
            _compute_cumfrac_terms_enrolled,
            term_rank_col="term_rank_core",
            min_student_term_rank_col="min_student_term_rank_core",
            cumnum_terms_enrolled_col="cumnum_core_terms_enrolled",
        ),
        cumfrac_noncore_terms_enrolled=ft.partial(
            _compute_cumfrac_terms_enrolled,
            term_rank_col="term_rank_noncore",
            min_student_term_rank_col="min_student_term_rank_noncore",
            cumnum_terms_enrolled_col="cumnum_noncore_terms_enrolled",
        ),
    )


def add_term_diff_features(
    df: pd.DataFrame,
    *,
    cols: list[str],
    max_term_num: int = 4,
    student_id_cols: str | list[str] = "student_id",
    term_num_col: str = "cumnum_terms_enrolled",
) -> pd.DataFrame:
    """
    Compute term-over-term differences per student for a set of columns, up to a specified
    maximum term number, where term numbers are _relative_, per student.

    Args:
        df
        cols
        max_term_num
        student_id_cols
        term_num_col
    """
    LOGGER.info("computing term diff features ...")
    student_id_cols = utils.types.to_list(student_id_cols)
    df_grped = (
        df.loc[:, student_id_cols + [term_num_col] + cols]
        .groupby(by=student_id_cols)
    )  # fmt: skip
    df_diffs = df.assign(
        **{f"{col}_diff_prev_term": df_grped[col].transform("diff") for col in cols}
    )
    df_pivots = []
    for col in cols:
        df_pivot_ = df_diffs.pivot(
            index=student_id_cols, columns=term_num_col, values=f"{col}_diff_prev_term"
        )
        # col 1 is always null (since no preceding periods to diff against)
        cols_to_drop = [1] + [
            col for col in df_pivot_.columns if int(col) > max_term_num
        ]
        df_pivots.append(
            df_pivot_.drop(columns=cols_to_drop)
            .rename(columns=lambda term_num: f"{col}_diff_term_{term_num - 1:.0f}_to_term_{term_num:.0f}")
        )  # fmt: skip
    df_pivot = pd.concat(df_pivots, axis="columns")
    return pd.merge(
        df_diffs, df_pivot, left_on=student_id_cols, right_index=True, how="left"
    )


def _compute_cumfrac_terms_enrolled(
    df: pd.DataFrame,
    *,
    term_rank_col: str = "term_rank",
    min_student_term_rank_col: str = "min_student_term_rank",
    cumnum_terms_enrolled_col: str = "cumnum_terms_enrolled",
) -> pd.Series:
    cumnum_terms_total = (df[term_rank_col] - df[min_student_term_rank_col]) + 1
    cumfrac_terms_enrolled = df[cumnum_terms_enrolled_col] / cumnum_terms_total
    return cumfrac_terms_enrolled.astype("Float32")


#######################
# DEPRECATED / BACKUP #
# only dig into these if databricks starts freaking out about expanding aggs func above

# def add_student_term_cumulative_features_databricks_hack(
#     df: pd.DataFrame,
#     *,
#     grp_cols: list[str],
#     sort_cols: list[str],
#     col_aggs: list[tuple[str, str | list[str]]],
# ) -> pd.DataFrame:
#     """
#     Compute features that aggregate cumulatively over each students' term history,
#     and assign them as additional columns.

#     Args:
#         df: Student-term features data, as output by :func:`make_student_term_dataset()` .
#         grp_cols: Columns needed to uniquely identify students,
#             used for grouping the rows in ``df`` .
#         sort_cols: Columns that sort rows within groups in chronological order,
#             such as term rank, term start dt, or "raw" (year, term) pairs.
#         col_aggs: Sequence of (column, agg-or-aggs) pairs, where each agg
#             is a pandas-compliant aggfunc.

#     Warning:
#         Under the hood, we've been forced to convert from standard pandas aggfuncs
#         into (a subset of) "transformation"-style cumulative methods. This is non-ideal,
#         but Databricks is behaving poorly -- and nondeterministically! -- when running
#         the unhacked sibling function above. So, in practice, the only supported aggs are
#         min, max, sum, and mean.

#     See Also:
#         - :func:`add_student_term_cumulative_features()`
#         - https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#built-in-transformation-methods
#     """
#     LOGGER.info("adding student-term cumulative features ...")
#     df_grped = (
#         df.sort_values(by=grp_cols + sort_cols, ignore_index=True)
#         # specifically *don't* re-sort when grouping
#         .groupby(by=grp_cols, observed=True, sort=False)
#     )

#     cumagg_to_cols = {
#         "cumcount": [sort_cols[0]],
#         "cummin": _get_cols_for_agg(col_aggs, "min"),
#         "cummax": _get_cols_for_agg(col_aggs, "max"),
#         "cumsum": (
#             _get_cols_for_agg(col_aggs, "sum")
#             + [
#                 col
#                 for col in df.columns
#                 if col.startswith(constants.NUM_COURSE_FEATURE_COL_PREFIX)
#                 and col != "num_courses_enrolled"
#             ]
#         ),
#         "cummean": (
#             _get_cols_for_agg(col_aggs, "mean")
#             + [
#                 col
#                 for col in df.columns
#                 if col.startswith(constants.FRAC_COURSE_FEATURE_COL_PREFIX)
#             ]
#         ),
#     }
#     dfs = []
#     for cumagg, cols in cumagg_to_cols.items():
#         if cumagg == "cumcount":
#             df_ = (
#                 df_grped[cols[0]]
#                 .transform("cumcount")
#                 .to_frame(name="cumnum_student_terms")
#             )
#         elif cumagg == "cummean":
#             df_ = (
#                 df_grped[cols]
#                 .expanding()
#                 .mean()
#                 .rename(columns={col: f"{col}_{cumagg}" for col in cols})
#                 .reset_index(level=list(range(len(grp_cols))), drop=True)
#             )
#         else:
#             df_ = (
#                 df_grped[cols]
#                 .transform(cumagg)
#                 .rename(columns={col: f"{col}_{cumagg}" for col in cols})
#             )
#         dfs.append(df_)

#     # make sure each cum-agg'd df has the same number of rows
#     assert len(set(len(df_) for df_ in dfs)) == 1

#     df_cfs = (
#         pd.concat(dfs, axis="columns")
#         .rename(
#             columns={
#                 "course_id_nunique_cummean": "nunique_courses_cummean",
#                 "course_cip_nunique_cummean": "nunique_course_subjects_cummean",
#             },
#         )
#         # let's also sneak in a derivative cumulative feature
#         .assign(
#             num_credits_earned_cumfrac=lambda df: (
#                 df["num_credits_earned_cumsum"] / df["num_credits_attempted_cumsum"]
#             )
#         )
#     )
#     return pd.concat([df, df_cfs], axis="columns")


# def _get_cols_for_agg(
#     col_aggs: list[tuple[str, str | list[str]]], agg: str
# ) -> list[str]:
#     return [
#         col_
#         for col_, agg_ in col_aggs
#         if (isinstance(agg_, str) and agg_ == agg) or agg in agg_
#     ]


# def compute_cumfrac_terms_unenrolled(
#     df: pd.DataFrame, *, grp_cols: list[str], term_rank_col: str
# ) -> pd.Series:
#     df_grped = df.groupby(by=grp_cols)
#     cumnum_terms_enrolled = df_grped[term_rank_col].cumcount() + 1
#     cumnum_terms_unenrolled = (
#         df_grped[term_rank_col]
#         .transform(lambda s: s.diff().sub(1.0).cumsum())
#         .fillna(0.0)
#         .astype("int64")
#     )
#     cumnum_terms_total = cumnum_terms_enrolled + cumnum_terms_unenrolled
#     cumfrac_terms_unenrolled = cumnum_terms_unenrolled.div(cumnum_terms_total)
#     return cumfrac_terms_unenrolled
