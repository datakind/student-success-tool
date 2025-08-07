import logging
import typing as t
import re

import pandas as pd
from pandas.api.types import (
    is_object_dtype,
    is_string_dtype,
    is_bool_dtype,
)

import h2o
from h2o.estimators.estimator_base import H2OEstimator
import shap

LOGGER = logging.getLogger(__name__)


def get_h2o_used_features(model: H2OEstimator) -> t.List[str]:
    """
    Extracts the actual feature names used by the H2O model (excluding dropped/constant columns).
    """
    # The last name is usually the response/target variable
    feature_names = model._model_json["output"]["names"][:-1]
    return list(feature_names)


def compute_h2o_shap_contributions(
    model: H2OEstimator,
    h2o_frame: h2o.H2OFrame,
    background_data: t.Optional[h2o.H2OFrame] = None,
    drop_bias: bool = True,
) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes SHAP-like contribution values from an H2O model.

    Args:
        model: Trained H2O model
        h2o_frame: h2o.H2OFrame for which to compute contributions
        background_data: Optional h2o.H2OFrame to use as the background reference for SHAP values
        drop_bias: Whether to exclude the 'BiasTerm' column

    Returns:
        contribs_df: SHAP contributions aligned with input features
        preprocessed_df: Input feature values used
    """
    used_features = get_h2o_used_features(model)
    hf_subset = h2o_frame[used_features]

    if background_data is not None:
        background_data = background_data[used_features]
        contribs_hf = model.predict_contributions(
            hf_subset, background_frame=background_data
        )
    else:
        contribs_hf = model.predict_contributions(hf_subset)

    contribs_df = contribs_hf.as_data_frame(use_pandas=True)
    preprocessed_df = hf_subset.as_data_frame(use_pandas=True)

    if drop_bias and "BiasTerm" in contribs_df.columns:
        contribs_df = contribs_df.drop(columns="BiasTerm")

    return contribs_df, preprocessed_df


def group_shap_values(
    df: pd.DataFrame,
    drop_bias_term: bool = False,
    group_missing_flags: bool = False,
) -> pd.DataFrame:
    """
    Groups one-hot encoded or exploded features into base features by summing.

    Args:
        df: DataFrame with SHAP contributions or one-hot encoded features.
        drop_bias_term: Whether to drop 'BiasTerm' column (only used for SHAP).
        group_missing_flags: Whether to group *_missing_flag columns with base features.

    Returns:
        grouped_df: DataFrame with values aggregated by base feature name.
    """
    if drop_bias_term and "BiasTerm" in df.columns:
        df = df.drop(columns=["BiasTerm"])

    grouped_data = {}

    for col in df.columns:
        base_col = get_base_feature_name(col, group_missing_flags)

        if base_col not in grouped_data:
            grouped_data[base_col] = df[col].copy()
        else:
            grouped_data[base_col] += df[col]

    return pd.DataFrame(grouped_data)


def group_feature_values(df: pd.DataFrame, group_missing_flags: bool) -> pd.DataFrame:
    """
    Groups one-hot encoded feature columns and *_missing_flag columns into base features.

    For categorical values, combines one-hot groups into readable values, handling missing flags.

    Args:
        df: Preprocessed input feature DataFrame (e.g., after one-hot encoding).
        group_missing_flags: Whether to group *_missing_flag into the same feature bucket.

    Returns:
        DataFrame with same number of rows, but fewer, grouped columns.
    """
    grouped: dict[str, list[int]] = {}

    for col in df.columns:
        base = get_base_feature_name(col, group_missing_flags)
        grouped.setdefault(base, []).append(df[col])

    out = {}
    for base, cols in grouped.items():
        if all(pd.api.types.is_numeric_dtype(c) for c in cols):
            out[base] = sum(cols)
        elif group_missing_flags:

            def resolve_row(values: list) -> str | None:
                val_names = [v for v in values if isinstance(v, str)]
                bool_flags = [v for v in values if isinstance(v, bool) and v]
                if bool_flags:
                    return "MISSING"
                if len(val_names) == 1:
                    return val_names[0]
                else:
                    raise ValueError(
                        f"Could not resolve base feature '{base}' due to ambiguous or missing encoding. "
                        f"Expected exactly one active one-hot value or missing flag. Found: {values}"
                    )

            stacked_df = pd.concat(cols, axis=1)
            out[base] = stacked_df.apply(lambda row: resolve_row(row.tolist()), axis=1)
        else:
            out[base] = pd.DataFrame(cols).apply(lambda row: None, axis=1)

    return pd.DataFrame(out)


def create_color_hint_features(
    original_df: pd.DataFrame, grouped_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Classifies each feature in the grouped input DataFrame as categorical or numeric,
    based on the original DataFrame's dtypes. Used for SHAP color hinting.

    Args:
        original_df (pd.DataFrame): The raw input DataFrame before encoding.
        grouped_df (pd.DataFrame): Input features post one-hot collapsing.

    Returns:
        pd.DataFrame: Same shape as grouped_df, with 'category' markers or original numeric values.
    """
    gray_features = pd.DataFrame(index=grouped_df.index)

    for col in grouped_df.columns:
        if col in original_df.columns:
            dtype = original_df[col].dtype
            is_categorical = (
                is_object_dtype(dtype)
                or isinstance(dtype, pd.CategoricalDtype)
                or is_string_dtype(dtype)
            ) and not is_bool_dtype(dtype)
        else:
            dtype = None
            is_categorical = False

        if is_categorical:
            gray_features[col] = "category"
            LOGGER.debug(f"{col}: classified as categorical (dtype={dtype})")
        else:
            gray_features[col] = grouped_df[col]
            LOGGER.debug(f"{col}: classified as numeric (dtype={dtype})")

    return gray_features


def get_base_feature_name(col: str, group_missing_flags: bool) -> str:
    """
    Derives the base feature name used for grouping SHAP values or input features.

    Args:
        col: Column name (possibly one-hot or missing flag)
        group_missing_flags: Whether to group missing flags with base feature

    Returns:
        Base feature name for grouping
    """
    if group_missing_flags and col.endswith("_missing_flag"):
        return col.removesuffix("_missing_flag")
    return re.split(r"\.", col, maxsplit=1)[0]


def plot_grouped_shap(
    contribs_df: pd.DataFrame,
    preprocessed_df: pd.DataFrame,
    original_df: pd.DataFrame,
    group_missing_flags: bool = False,
) -> None:
    """
    Plot grouped SHAP values as a global summary plot. One-hot encoded features are grouped under their base feature name.
    Missingness flags are optionally grouped with their base feature based on the `group_missing_flags` flag.
    A color hint matrix is built from the raw data to improve SHAP summary plot interpretability.

    Parameters:
        contribs_df: DataFrame of SHAP contributions (from H2O), including one-hot or exploded categorical features.
        preprocessed_df: Preprocessed feature matrix (e.g., after imputation and one-hot encoding), matching SHAP columns.
        original_df: Original raw input DataFrame (before preprocessing), used for inferring data types and color hints.
        group_missing_flags: Whether to group missingness flag columns (e.g., 'math_placement_missing_flag')
                             into their corresponding base feature (e.g., 'math_placement') in the SHAP plot.
    """
    grouped_shap = group_shap_values(
        contribs_df, group_missing_flags=group_missing_flags
    )
    grouped_inputs = group_feature_values(
        preprocessed_df, group_missing_flags=group_missing_flags
    )
    color_hint = create_color_hint_features(original_df, grouped_inputs)

    shap.summary_plot(
        grouped_shap.values,
        features=color_hint,
        feature_names=grouped_shap.columns,
    )
