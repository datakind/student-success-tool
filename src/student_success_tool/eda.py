import itertools
import logging
import typing as t

import numpy as np
import pandas as pd
import scipy.stats as ss

from . import utils

LOGGER = logging.getLogger(__name__)


def assess_unique_values(data: pd.DataFrame, cols: str | list[str]) -> dict[str, int]:
    """
    Assess unique values in ``data`` given by the combination of columns in ``cols`` ,
    including counts of nunique, duplicates, and nulls.

    Args:
        data
        cols
    """
    unique_data = data.loc[:, utils.types.to_list(cols)]
    is_duplicated = unique_data.duplicated()
    return {
        "num_uniques": is_duplicated.eq(False).sum(),
        "num_dupes": is_duplicated.sum(),
        "num_with_null_values": unique_data.isna().sum(axis="columns").gt(0).sum(),
    }


def compute_summary_stats(
    data: pd.DataFrame,
    *,
    include: t.Optional[str | list[str]] = None,
    exclude: t.Optional[str | list[str]] = None,
    percentiles: t.Optional[list[float]] = None,
) -> pd.DataFrame:
    """
    Compute summary stats for columns in ``data`` matching one or multiple dtypes
    using standard :meth:`pd.DataFrame.describe()` , supplemented with null count/pct.

    Args:
        data
        include: One or multiple dtypes whose columns will be included in result.
        exclude: One or multiple dtypes whose columns will be excluded from result.
        percentiles: Percentiles to include in result, given as floats between 0 and 1.

    References:
        - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.select_dtypes.html
        - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html
    """
    num_rows = data.shape[0]
    include = utils.types.to_list(include) if include is not None else None
    exclude = utils.types.to_list(exclude) if exclude is not None else None
    data_selected = data.select_dtypes(include=include, exclude=exclude)  # type: ignore
    data_described = data_selected.describe(percentiles=percentiles).T.assign(
        null_count=data_selected.isna().sum(),
        null_pct=lambda df: (100 * df["null_count"] / num_rows).round(decimals=1),
    )
    return data_described


def compute_group_counts_pcts(
    data: pd.DataFrame,
    cols: str | list[str],
    *,
    sort: bool = True,
    ascending: bool = False,
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Compute value counts and percent totals in ``data`` for groups defined by ``cols`` .

    Args:
        data
        cols
        sort
        ascending
        dropna

    References:
        - https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.value_counts.html
    """
    return (
        # data.value_counts(cols, dropna = False) for some reason still drops NaNs. We use
        # data[cols].value_counts() to avoid this error
        data[cols]
        .value_counts(sort=sort, ascending=ascending, dropna=dropna)
        .to_frame(name="count")
        .assign(
            pct=lambda df: (100 * df["count"] / df["count"].sum()).round(decimals=1)
        )
    )


def compute_crosstabs(
    data: pd.DataFrame,
    index_cols: str | list[str],
    column_cols: str | list[str],
    value_col: t.Optional[str] = None,
    aggfunc: t.Optional[t.Callable] = None,
    margins: bool = True,
    normalize: bool | t.Literal["all", "index", "columns"] = False,
) -> pd.DataFrame:
    """
    Args:
        data
        index_cols
        column_cols
        value_col
        aggfunc
        margins
        normalize

    References:
        - https://pandas.pydata.org/docs/reference/api/pandas.crosstab.html
    """
    index = (
        data[index_cols]
        if isinstance(index_cols, str)
        else [data[col] for col in index_cols]
    )
    columns = (
        data[column_cols]
        if isinstance(column_cols, str)
        else [data[col] for col in column_cols]
    )
    values = None if value_col is None else data[value_col]
    ct = pd.crosstab(
        index,
        columns,
        values=values,  # type: ignore
        aggfunc=aggfunc,  # type: ignore
        margins=margins,
        normalize=normalize,
    )
    assert isinstance(ct, pd.DataFrame)  # type guard
    if normalize is not False:
        ct = ct.round(decimals=3)
    return ct


def compute_pairwise_associations(
    df: pd.DataFrame,
    *,
    ref_col: t.Optional[str] = None,
    exclude_cols: t.Optional[str | list[str]] = None,
) -> pd.DataFrame:
    """
    Compute pairwise associations between all columns and each other or, instead,
    all columns and a specified reference column.

    Per-pair association metrics depend on the data types of each:

        - nominal-nominal => Cramer's V
        - numeric-numeric => Spearman rank correlation
        - nominal-numeric => Correlation ratio

    Args:
        df
        ref_col: Reference column against which associations are to be computed.
            If None, all pairwise associations are computed.
        exclude_cols: One or multiple columns to exclude from computing associations;
            for example, if values are unique identifiers or all a single value,
            making their associations irrelevant.

    References:
        - https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
        - https://en.wikipedia.org/wiki/Correlation_ratio
    """
    # cast datetime columns to numeric, boolean to string
    df = df.assign(
        **{
            col: pd.to_numeric(df[col])
            for col in df.select_dtypes(include="datetime").columns
        }
        | {
            col: df[col].astype("string")
            for col in df.select_dtypes(include="boolean").columns
        }
    )
    # identify and organize columns in df
    if exclude_cols:
        df = df.drop(columns=exclude_cols)
    cols = df.columns.tolist()
    nominal_cols = set(
        df.select_dtypes(include=["category", "string", "boolean"]).columns.tolist()
    )
    numeric_cols = set(df.select_dtypes(include="number").columns.tolist())
    single_value_cols = _get_single_value_columns(df)
    # store col-col association values
    ref_cols = cols if ref_col is None else [ref_col]
    df_assoc = pd.DataFrame(index=cols, columns=ref_cols, dtype="Float32")
    for col1, col2 in itertools.product(cols, ref_cols):
        if not pd.isna(df_assoc.at[col1, col2]):
            continue

        is_symmetric = False
        if col1 == col2:  # self-association
            assoc = 1.0
        elif col1 in single_value_cols or col2 in single_value_cols:  # n/a
            assoc = None
        elif col1 in nominal_cols and col2 in nominal_cols:  # nom-nom
            assoc = _cramers_v(df[col1], df[col2])
            is_symmetric = True
        elif (col1 in nominal_cols and col2 in numeric_cols) or (
            col1 in numeric_cols and col2 in nominal_cols
        ):  # nom-num
            assoc = _correlation_ratio(df[col1], df[col2])
        elif col1 in numeric_cols and col2 in numeric_cols:  # num-num
            assoc = df[col1].corr(df[col2], method="spearman")
            is_symmetric = True
        else:
            LOGGER.warning(
                "'%s' and/or '%s' columns' dtypes (%s and/or %s) aren't supported "
                "for association computation; skipping ...",
                col1,
                col2,
                df[col1].dtype,
                df[col2].dtype,
            )
            assoc = None

        df_assoc.loc[col1, col2] = assoc
        if is_symmetric and len(ref_cols) > 1:
            df_assoc.loc[col2, col1] = assoc
        if assoc is not None:
            LOGGER.debug("%s – %s association = %s", col1, col2, assoc)
    return df_assoc


def _get_single_value_columns(df: pd.DataFrame) -> set[str]:
    sv_cols = []
    for col in df.columns:
        try:
            nunique = df[col].nunique()
        except TypeError:  # womp
            continue
        if nunique == 1:
            sv_cols.append(col)
    return set(sv_cols)


def _cramers_v(s1: pd.Series, s2: pd.Series) -> float | None:
    """
    Compute Cramer's V statistic for nominal-nominal association,
    which is symmetric -- i.e. V(x, y) == V(y, x).

    References:
        - https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V

    See Also:
        - :func:`scipy.stats.contingency.association()`
    """
    if not pd.api.types.is_string_dtype(s1) or not pd.api.types.is_string_dtype(s2):
        raise ValueError()

    s1, s2 = _drop_incomplete_pairs(s1, s2)
    if s1.empty or s2.empty:
        return None

    confusion_matrix = pd.crosstab(s1, s2)
    correction = False if confusion_matrix.shape[0] == 2 else True
    try:
        result = ss.contingency.association(
            confusion_matrix, method="cramer", correction=correction
        )
        assert isinstance(result, float)
        return result
    except ValueError:
        return None


def _correlation_ratio(s1: pd.Series, s2: pd.Series) -> float | None:
    """
    Compute the Correlation Ratio for nominal-numeric association.

    References:
        - https://en.wikipedia.org/wiki/Correlation_ratio

    Note:
        ``s1`` and ``s2`` are automatically detected as being categorical or numeric,
        and handled correspondingly in the calculations.
    """
    if pd.api.types.is_string_dtype(s1) and pd.api.types.is_numeric_dtype(s2):
        categories = s1
        measurements = s2
    elif pd.api.types.is_numeric_dtype(s1) and pd.api.types.is_string_dtype(s2):
        categories = s2
        measurements = s1
    else:
        raise ValueError()

    categories, measurements = _drop_incomplete_pairs(categories, measurements)
    if categories.empty or measurements.empty:
        return None

    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat) + 1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
    numerator = np.sum(
        np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2))
    )
    denominator = np.sum(np.power(np.subtract(measurements, y_total_avg), 2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator / denominator)
    return eta


def _drop_incomplete_pairs(s1: pd.Series, s2: pd.Series) -> tuple[pd.Series, pd.Series]:
    df = pd.DataFrame({"s1": s1, "s2": s2})
    df = df.dropna(axis="index", how="any", ignore_index=True)  # type: ignore
    return (df["s1"], df["s2"])
