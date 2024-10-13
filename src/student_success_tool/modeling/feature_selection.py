import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor


def select_features(
    train_df,
    not_features_cols,
    force_include_cols,
    incomplete_threshold=0.5,
    low_variance_threshold=0.0,
):
    """Select features by dropping incomplete features, low variance features,
    and variables with high correlations to others.

    Args:
        train_df (pd.DataFrame): dataframe of features to select
        not_features_cols (list[str]): list of column names that are not features and should not
            be run through the feature selection algorithm. For example, demographics, IDs, outcome variables, etc.
        force_include_cols (list[str]): list of features to force include in the final dataset.
        incomplete_threshold (float, optional): threshold for level of incompleteness. Defaults to 0.5.
        low_variance_threshold (float, optional): threshold for level of low variance. Defaults to 0.0.

    Returns:
        pd.DataFrame: train_df with not_features_cols, force_include_cols, and any other columns selected
            by the algorithms
    """

    X = train_df.drop(columns=not_features_cols + force_include_cols)

    ###################################
    ##### Unsupervised selection ######
    ###################################

    # incomplete features - features with lots of nulls
    # Note: we should create dummy variables prior to running this process. If one of the dummies represents NaN, we won't drop the categoricals that have lots of nulls, but instead can capture any patterns that may relate to the nulls. If there are non-dummy categorical variables that get dropped here, consider pre-processing into dummy variables for this reason.
    X = drop_incomplete_features(
        X, threshold=incomplete_threshold
    )  # TODO: tune this threshold over time

    # features with low variance
    # Note: only removing features with 0 variance for now, as features with low
    # but non-zero variance may be powerful in predicting the outcome
    X = drop_low_variance_features(X, threshold=low_variance_threshold)

    # multi-collinearity: it may not interfere with the model's performance, but it does negatively impact the interpretation of the predictors
    features_force_include_df = train_df[
        list(set(list(X.columns.values) + force_include_cols))
    ]
    selected_features_df = drop_collinear_features_iteratively(
        features_force_include_df, force_include_cols
    )

    selected_cols = set(list(selected_features_df.columns.values)) - set(
        force_include_cols
    )
    print(f"Original N features: {train_df.shape[1]}")
    print(f"Selected {len(selected_cols)} features: {selected_cols}")

    keep_cols = list(selected_cols)
    additional_keep_cols = not_features_cols + force_include_cols
    keep_cols.extend(additional_keep_cols)
    print(
        f"Plus {len(additional_keep_cols)} demographic, outcome, and force-included columns: {additional_keep_cols}"
    )
    selected_features_df = train_df[keep_cols]
    return selected_features_df


def drop_incomplete_features(features, threshold):
    """Drop columns from dataframe that have at least some fraction of nulls.

    Args:
        features (pd.DataFrame): dataframe of columns to assess nulls across
        threshold (float): fraction of nulls deemed unacceptable. Any columns in
            the features dataframe with this fraction of nulls or more will be dropped.

    Returns:
        pd.DataFrame: features data without the incomplete columns
    """
    pct_null = features.isna().sum() / features.shape[0]
    incomplete_features = pct_null[pct_null >= threshold].index
    if len(incomplete_features) > 0:
        print(f"Dropping incomplete features: {incomplete_features}")
        return features.drop(columns=incomplete_features)
    else:
        print(f"No features with at least {threshold} proportion of nulls.")
        return features


def drop_low_variance_features(features, threshold):
    """Drop columns with low or no variance, according to a threshold.

    Args:
        features (pd.DataFrame): dataframe of columns to assess variance
        threshold (float): Features with a training-set variance lower than this threshold will be removed.

    Returns:
        pd.DataFrame: features data without the low variance columns
    """
    selector = VarianceThreshold(threshold=threshold)
    numeric_features = features.select_dtypes(include="number")

    column_contains_inf = np.isinf(numeric_features).any()
    if column_contains_inf.any():
        print("The following columns contain infinity. Remove and try again!")
        print(list(numeric_features.columns[column_contains_inf]))
        raise Exception
    selector.fit(numeric_features)
    no_variance_features = numeric_features.columns.values[~selector.get_support()]
    if (n := len(no_variance_features)) > 0:
        print(
            f"Dropping {n} low variance (<={threshold}) features: {no_variance_features}"
        )
        return features.drop(columns=no_variance_features)
    else:
        print(f"No features with less than {threshold} variance.")
        return features


def drop_collinear_features_iteratively(features, force_include_cols):
    """Use Variance Inflation Factor (VIF) to drop collinear features iteratively.
    The function takes the following steps:
    1. Selects only numeric features for VIF analysis
    2. Impute missing values - this is required by the variance_inflation_factor()
    function. Because we typically rely on default AutoML strategies for imputation,
    we expect to have null values in our data. TODO: we may want to revisit this
    decision if we want more consistent imputation between the VIF analysis and the
    modeling, perhaps deciding to do our own imputation prior to this VIF analysis.
    3. Find the highest VIF. If it is over 10, it is considered too high and the
    feature is contributing to multicollinearity - we drop this feature.
    4. Repeat step 3 with the remaining features until there are no VIF > 10.

    Note: the function used, variance_inflation_factor(), does not take an
    intercept into account. See: https://github.com/statsmodels/statsmodels/issues/2376
    If we want to use a centered VIF rather than an uncentered VIF, we would need to add
    a constant columns to the features dataframe and then re-calculate VIF.

    Troubleshooting: if this hangs, try disabling mlflow logging:
        import mlflow
        mlflow.autolog(disable=True)

    Args:
        features (pd.DataFrame): dataframe of features to assess for multicollinearity

    Returns:
        pd.DataFrame: features not considered collinear according to a VIF threshold of 10
    """

    np.seterr(divide="ignore", invalid="ignore")

    numeric_features = features.select_dtypes(include="number")
    imp = SimpleImputer(missing_values=np.nan, strategy="mean")
    numeric_features_imputed = pd.DataFrame(imp.fit_transform(numeric_features))
    numeric_features_imputed.columns = numeric_features.columns.values

    # A VIF of 1 indicates the feature has no correlation with any of the other features. A VIF exceeding 10 is too high and contributing to multicollinearity
    n_features_dropped_so_far = 0
    while (
        max_vif := max(
            (
                uncentered_vif_dict := {
                    # calculate VIF of features that are not force-included, against all numeric variables
                    col: variance_inflation_factor(numeric_features_imputed, col_index)
                    for col_index, col in enumerate(
                        numeric_features_imputed.columns.values
                    )
                    if col not in force_include_cols
                }
            ).values()
        )
    ) > 10:
        highest_vif_cols = [
            col for col, vif in uncentered_vif_dict.items() if vif == max_vif
        ]
        n_drop_this_round = len(highest_vif_cols)
        n_features_dropped_so_far += n_drop_this_round
        print(f"Dropping {n_drop_this_round} column(s) with VIF {max_vif}")
        print(highest_vif_cols)
        features = features.drop(columns=highest_vif_cols)
        numeric_features_imputed = numeric_features_imputed.drop(
            columns=highest_vif_cols
        )
    print(f"Dropped {n_features_dropped_so_far} collinear features.")
    assert all(
        [col in features.columns for col in force_include_cols]
    ), "The dataset with selected features is missing one of the force include variables!"
    return features
