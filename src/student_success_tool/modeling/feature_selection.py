import logging
import typing as t
from collections.abc import Collection

import numpy as np
import pandas as pd
import sklearn.feature_selection
import sklearn.impute
from statsmodels.stats.outliers_influence import variance_inflation_factor

LOGGER = logging.getLogger(__name__)


def select_features(
    df: pd.DataFrame,
    *,
    non_feature_cols: t.Optional[list[str]] = None,
    force_include_cols: t.Optional[list[str]] = None,
    incomplete_threshold: float = 0.5,
    low_variance_threshold: float = 0.0,
    collinear_threshold: t.Optional[float] = 10.0,
) -> pd.DataFrame:
    """
    Select features by dropping incomplete features, low variance features,
    and variables with high correlations to others.

    Args:
        df: dataframe of features to select
        non_features_cols: list of column names that are not features and should not
            be run through the feature selection algorithm. For example, demographics, IDs, outcome variables, etc.
        force_include_cols: list of features to force include in the final dataset.
        incomplete_threshold: Threshold for determining incomplete features.
        low_variance_threshold: Threshold for determining low-variance features.
        collinear_threshold: Threshold for determining collinear features;
            if null, skip this selection step.

    Returns:
        df with non_features_cols, force_include_cols, and any other columns selected
        by the algorithms
    """
    LOGGER.info("selecting features ...")
    non_feature_cols = non_feature_cols or []
    force_include_cols = force_include_cols or []
    df_selected = (
        # we'll add these columns back in later
        df.drop(columns=non_feature_cols + force_include_cols)
        # NOTE: we should create dummy variables for categoricals prior to running this
        # if one of the dummies represents NaN, we won't drop the categoricals that have lots of nulls,
        # but instead can capture any patterns that may relate to the nulls.
        # If there are non-dummy categorical variables that get dropped here,
        # consider pre-processing into dummy variables for this reason.
        # TODO: tune this threshold over time?
        .pipe(drop_incomplete_features, threshold=incomplete_threshold)
        # NOTE: only removing features with variance == 0.0 by default
        # since features with low but non-zero variance may still be predictive
        .pipe(drop_low_variance_features, threshold=low_variance_threshold)
    )
    sel_incl_feature_cols = list(set(df_selected.columns.tolist() + force_include_cols))
    df_selected = df.loc[:, sel_incl_feature_cols]
    if collinear_threshold is not None:
        # multi-collinearity: it may not interfere with the model's performance
        # but it does negatively affect the interpretation of the predictors
        df_selected = drop_collinear_features_iteratively(
            df_selected,
            threshold=collinear_threshold,
            force_include_cols=force_include_cols,
        )

    orig_feature_cols = set(df.columns) - set(non_feature_cols)
    selected_feature_cols = set(df_selected.columns)
    keep_cols = set(non_feature_cols) | selected_feature_cols
    LOGGER.info(
        "selected %s out of %s (%s%%) feature columns: %s",
        len(selected_feature_cols),
        len(orig_feature_cols),
        round(100.0 * len(selected_feature_cols) / len(orig_feature_cols), 1),
        selected_feature_cols,
    )
    # maintain the original column order, in case that's meaningful
    return df.loc[:, df.columns.isin(keep_cols)]


def drop_incomplete_features(
    df: pd.DataFrame, *, threshold: float = 0.5
) -> pd.DataFrame:
    """
    Drop columns from dataframe that have at least ``threshold`` fraction of null values.

    Args:
        df
        threshold: Fraction of null values above which columns are deemed "incomplete"
            and dropped from ``df`` .

    Returns:
        ``df`` without the incomplete feature columns
    """
    frac_null = df.isna().sum(axis="index") / df.shape[0]
    incomplete_cols = frac_null.loc[frac_null.ge(threshold)].index.tolist()
    if incomplete_cols:
        LOGGER.info(
            "dropping %s incomplete features: %s",
            len(incomplete_cols),
            incomplete_cols,
        )
        return df.drop(columns=incomplete_cols)
    else:
        LOGGER.info("no features found with a fraction of null values >= %s", threshold)
        return df


def drop_low_variance_features(
    df: pd.DataFrame, *, threshold: float = 0.0
) -> pd.DataFrame:
    """
    Drop columns with low or no variance, according to a threshold.

    Args:
        df
        threshold: Variance of values below which columns are dropped from ``df`` .

    Returns:
        ``df`` without the low-variance columns
    """
    df_numeric = df.select_dtypes(include="number")
    column_contains_inf = np.isinf(df_numeric).any()
    if column_contains_inf.any():
        inf_cols = df_numeric.columns[column_contains_inf].tolist()  # type: ignore
        LOGGER.error(
            "Columns %s contain infinite values -- remove and try again!", inf_cols
        )
        raise ValueError()

    selector = (
        sklearn.feature_selection.VarianceThreshold(threshold=threshold)
        .fit(df_numeric)
    )  # fmt: skip
    df_categorical = df.select_dtypes(include=["string", "category", "boolean"])
    constant_value_cols = {
        col
        for col, nunique in df_categorical.nunique().eq(1).to_dict().items()
        if nunique == 1
    }
    low_variance_cols = list(
        set(df_numeric.columns) - set(selector.get_feature_names_out())
        | constant_value_cols
    )
    if low_variance_cols:
        LOGGER.info(
            "dropping %s low-variance features: %s",
            len(low_variance_cols),
            low_variance_cols,
        )
        return df.drop(columns=low_variance_cols)
    else:
        LOGGER.info("no features found with variance < %s", threshold)
        return df


def drop_collinear_features_iteratively(
    df: pd.DataFrame,
    *,
    threshold: float = 10.0,
    force_include_cols: t.Optional[Collection[str]] = None,
) -> pd.DataFrame:
    """
    Use Variance Inflation Factor (VIF) to drop collinear features iteratively.

    The function takes the following steps:
    1. Selects only numeric features for VIF analysis
    2. Impute missing values - this is required by the variance_inflation_factor()
    function. Because we typically rely on default AutoML strategies for imputation,
    we expect to have null values in our data. TODO: we may want to revisit this
    decision if we want more consistent imputation between the VIF analysis and the
    modeling, perhaps deciding to do our own imputation prior to this VIF analysis.
    3. Find the highest VIF. If it is over threshold, it is considered too high and the
    feature is contributing to multicollinearity - we drop this feature.
    4. Repeat step 3 with the remaining features until there are no VIF >= threshold.

    Note: the function used, variance_inflation_factor(), does not take an
    intercept into account. See: https://github.com/statsmodels/statsmodels/issues/2376
    If we want to use a centered VIF rather than an uncentered VIF, we would need to add
    a constant columns to the features dataframe and then re-calculate VIF.

    Troubleshooting: if this hangs, try disabling mlflow logging:
        import mlflow
        mlflow.autolog(disable=True)

    Args:
        df
        threshold: Variance Inflaction Factor (VIF) above which columns are considered
            "multicollinear" and dropped from ``df`` . A VIF of 1 indicates no correlation
            with any other feature; a VIF of 10 is considered "very high" and contributing
            to multicollinearity.
        force_include_cols

    Returns:
        features not considered collinear according to a VIF threshold
    """
    np.seterr(divide="ignore", invalid="ignore")
    force_include_cols = force_include_cols or []

    numeric_df = df.select_dtypes(include=["number"])
    bool_df = df.select_dtypes(include=["boolean"]).astype("Int64")

    if numeric_df.empty and bool_df.empty:
        LOGGER.warning("no numeric columns found, so no collinear features to drop")
        return df

    imputer: sklearn.impute.SimpleImputer = sklearn.impute.SimpleImputer(
        missing_values=np.nan, strategy="mean"
    ).set_output(transform="pandas")  # type: ignore
    df_num_imputed = imputer.fit_transform(numeric_df)
    assert isinstance(df_num_imputed, pd.DataFrame)  # type guard

    df_features = df_num_imputed

    n_features_dropped_so_far = 0

    if not bool_df.empty:
        bool_imputer: sklearn.impute.SimpleImputer = sklearn.impute.SimpleImputer(
            missing_values=np.nan, strategy="most_frequent"
        ).set_output(transform="pandas")  # type: ignore
        df_bool_imputed = bool_imputer.fit_transform(bool_df)
        assert isinstance(df_bool_imputed, pd.DataFrame)  # type guard
        df_features = pd.concat([df_features, df_bool_imputed], axis=1)
        # drop if there are any boolean columns perfectly duplicate of the numeric cols
        duplicated_cols = df_features.columns[
            df_features.T.duplicated(keep="first")
        ].tolist()
        df_features = df_features.drop(columns=duplicated_cols)
        df = df.drop(columns=duplicated_cols)
        n_features_dropped_so_far += len(duplicated_cols)

    print(df_features.columns.tolist())
    print(df_features.dtypes)

    # calculate initial VIFs for features that aren't force-included
    uncentered_vif_dict = {
        col: variance_inflation_factor(df_features, col_idx)
        for col_idx, col in enumerate(df_features.columns)
        if col not in force_include_cols
    }
    if np.isinf(list(uncentered_vif_dict.values())).all():
        LOGGER.warning(
            "all features are perfectly correlated with one another; "
            "no collinear features will be dropped ..."
        )
        return df

    while (max_vif := max(uncentered_vif_dict.values())) >= threshold:
        highest_vif_cols = [
            col for col, vif in uncentered_vif_dict.items() if vif >= max_vif
        ]
        n_features_dropped_so_far += len(highest_vif_cols)
        LOGGER.info(
            "dropping %s columns with VIF >= %s: %s ...",
            len(highest_vif_cols),
            threshold,
            highest_vif_cols,
        )
        df = df.drop(columns=highest_vif_cols)
        df_features = df_features.drop(columns=highest_vif_cols)

        # recalculate VIFs after dropping columns
        uncentered_vif_dict = {
            col: variance_inflation_factor(df_features, col_idx)
            for col_idx, col in enumerate(df_features.columns)
            if col not in force_include_cols
        }

    LOGGER.info("dropping %s collinear features", n_features_dropped_so_far)

    if not all([col in df.columns for col in force_include_cols]):
        raise ValueError(
            "The dataset with selected features is missing one of the force include variables!"
        )

    return df

    # TODO: figure out why this below code gives different results from the original :/
    # collinear_cols = []
    # while True:
    #     candidate_cols = [
    #         col for col in df_imputed.columns if col not in force_include_cols
    #     ]
    #     if not candidate_cols:
    #         break

    #     col_vifs = {
    #         col: variance_inflation_factor(df_imputed, col_index)
    #         for col_index, col in enumerate(candidate_cols)
    #     }
    #     max_vif = max(col_vifs.values())
    #     if max_vif < threshold:
    #         break

    #     max_vif_cols = [col for col, vif in col_vifs.items() if vif == max_vif]
    #     LOGGER.info(
    #         "dropping %s columns with VIF >= %s: %s ...",
    #         len(max_vif_cols),
    #         threshold,
    #         max_vif_cols,
    #     )
    #     df_imputed = df_imputed.drop(columns=max_vif_cols)
    #     collinear_cols.extend(max_vif_cols)

    # if collinear_cols:
    #     LOGGER.info(
    #         "dropping %s collinear features: %s",
    #         len(collinear_cols),
    #         collinear_cols,
    #     )
    #     return df.drop(columns=collinear_cols)
    # else:
    #     LOGGER.info("no collinear features found with VIF >= %s", threshold)
    #     return df
