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
    return feature_names


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
        input_df: Input feature values used
    """
    used_features = get_h2o_used_features(model)
    hf_subset = h2o_frame[used_features]

    if background_data is not None:
        background_data = background_data[used_features]
        contribs_hf = model.predict_contributions(
            hf_subset, background_data=background_data
        )
    else:
        contribs_hf = model.predict_contributions(hf_subset)

    contribs_df = contribs_hf.as_data_frame(use_pandas=True)
    input_df = hf_subset.as_data_frame(use_pandas=True)

    if drop_bias and "BiasTerm" in contribs_df.columns:
        contribs_df = contribs_df.drop(columns="BiasTerm")

    return contribs_df, input_df


def group_shap_by_feature(contribs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Group SHAP contributions by original feature name by aggregating
    one-hot encoded components (e.g., 'feature.value') back to 'feature'.

    Args:
        contribs_df: DataFrame of SHAP contributions, with optional 'BiasTerm'

    Returns:
        grouped_df: DataFrame with SHAP values summed by original feature name
    """
    if "BiasTerm" in contribs_df.columns:
        contribs_df = contribs_df.drop(columns=["BiasTerm"])

    grouped_data = {}
    for col in contribs_df.columns:
        # Extract base feature name before the first dot
        base_col = re.split(r"\.", col, maxsplit=1)[0]
        if base_col not in grouped_data:
            grouped_data[base_col] = contribs_df[col].copy()
        else:
            grouped_data[base_col] += contribs_df[col]

    grouped_df = pd.DataFrame(grouped_data)
    return grouped_df


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


def group_feature_values_by_feature(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups one-hot encoded input feature values back to base features
    by summing over the components (same logic as group_shap_by_feature).

    Args:
        input_df: pandas DataFrame with one-hot columns

    Returns:
        grouped_df: pandas DataFrame with grouped input values
    """
    grouped_data = {}
    for col in input_df.columns:
        base_col = re.split(r"\.", col, maxsplit=1)[0]
        if base_col not in grouped_data:
            grouped_data[base_col] = input_df[col].copy()
        else:
            grouped_data[base_col] += input_df[col]
    return pd.DataFrame(grouped_data)


def plot_grouped_shap(
    contribs_df: pd.DataFrame, input_df: pd.DataFrame, original_df: pd.DataFrame
) -> None:
    """
    Plot grouped shap values based on contributions dataframe (shap values), input dataframe, which
    contain the one-hot encoding columns, and the original dataframe, which was the data used for training. This
    dataframe is used purely for color hint and dtypes.

    The output of this will be a global shap plot for each feature in the model ranked top to bottom
    in terms of importance.
    """
    grouped_shap = group_shap_by_feature(contribs_df)
    grouped_inputs = group_feature_values_by_feature(input_df)
    color_hint = create_color_hint_features(original_df, grouped_inputs)

    shap.summary_plot(
        grouped_shap.values, features=color_hint, feature_names=grouped_shap.columns
    )
