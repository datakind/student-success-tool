import logging
import typing as t
import mlflow
import re
import math
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import (
    is_object_dtype,
    is_string_dtype,
    is_bool_dtype,
    CategoricalDtype,
    pandas_dtype,
)

import h2o
from h2o.estimators.estimator_base import H2OEstimator
import shap

from . import utils

LOGGER = logging.getLogger(__name__)


def get_h2o_used_features(model: H2OEstimator) -> t.List[str]:
    """
    Extracts the actual feature names used by the H2O model (excluding dropped/constant columns).
    """
    out = model._model_json["output"]
    params = model.actual_params

    names = list(out["names"])

    # Figure out the response/target name
    response = (
        (out.get("response_column") or {}).get("name")
        or params.get("response_column")
        or params.get("y")
    )

    # Collect special (non-predictor) columns to drop
    non_predictors = set()
    if response:
        non_predictors.add(response)

    for k in ("weights_column", "offset_column", "fold_column"):
        v = params.get(k)
        if isinstance(v, dict):
            v = v.get("column_name")
        if v:
            non_predictors.add(v)

    # Keep only real predictors
    return [c for c in names if c not in non_predictors]


def predict_probs_h2o(
    features: pd.DataFrame | np.ndarray,
    model: H2OEstimator,
    *,
    feature_names: t.Optional[list[str]] = None,
    pos_label: t.Optional[bool | str] = None,
    dtypes: t.Optional[dict[str, object]] = None,
) -> np.ndarray:
    """
    Predict target probabilities using an H2O model.
    """
    if isinstance(features, np.ndarray):
        if feature_names is None:
            raise ValueError("feature_names must be provided when using a numpy array.")
        features = pd.DataFrame(features, columns=feature_names)

    if dtypes:
        features = features.astype(dtypes)

    h2o_features = h2o.H2OFrame(features)
    pred = utils._to_pandas(model.predict(h2o_features))

    if pos_label is not None:
        pos_label_str = str(pos_label)
        if pos_label_str not in pred.columns:
            raise ValueError(
                f"pos_label {pos_label_str} not found in prediction output columns: {pred.columns}"
            )
        return np.array(pred[pos_label_str].values)
    else:
        prob_cols = [col for col in pred.columns if col != "predict"]
        return np.array(pred[prob_cols].values)


def predict_contribs_batched(
    model: H2OEstimator,
    hf: h2o.H2OFrame,
    *,
    batch_rows: int = 1000,
    top_n: t.Optional[int] = None,
    bottom_n: int = 0,
    compare_abs: bool = True,
    output_format: t.Optional[str] = None,
    background_frame: t.Optional[h2o.H2OFrame] = None,
    drop_bias: bool = True,
    output_space: bool = True,
) -> pd.DataFrame:
    """
    Compute SHAP/TreeSHAP contributions in batches and return a combined DataFrame.

    Contributions are computed in row batches to reduce memory usage. Each batch is
    immediately converted to pandas, then concatenated at the end.

    Args:
        model: Trained H2O model.
        hf: Input H2OFrame with features to score.
        batch_rows: Maximum number of rows per batch. Defaults to 1000.
        top_n: Return only the top N features by contribution. Defaults to None.
        bottom_n: Return only the bottom N features by contribution. Defaults to 0.
        compare_abs: Rank features by absolute contribution. Defaults to True.
        output_format: Format for output, e.g. "Compact" for XGBoost. Defaults to None.
        background_frame: Optional reference data for SHAP baseline. Defaults to None.
        drop_bias: If True, drop the BiasTerm column. Defaults to True.
        output_space: If True, return contributions in the model’s response
            space (e.g., probabilities). If False, keep logit space. Defaults to True.

    Returns:
        pd.DataFrame: Concatenated contributions aligned to rows of `hf`.
    """
    n = hf.nrows
    batches = max(1, math.ceil(n / batch_rows))
    dfs: t.List[pd.DataFrame] = []

    # Build kwargs once
    kwargs: dict = {}
    if top_n is not None:
        kwargs.update(dict(top_n=top_n, bottom_n=bottom_n, compare_abs=compare_abs))
    if output_format is not None:
        kwargs.update(dict(output_format=output_format))
    if background_frame is not None:
        kwargs.update(dict(background_frame=background_frame))
    if output_space:
        kwargs.update(dict(output_space=True))

    LOGGER.info(
        f"Starting SHAP (with per-batch pandas conversion): {n} rows, {batches} batches of up to {batch_rows}"
    )

    for b in range(batches):
        start = b * batch_rows
        end = min((b + 1) * batch_rows, n)
        chunk = hf[start:end, :]  # lightweight slice
        t0 = time.time()
        contrib_chunk_hf = model.predict_contributions(chunk, **kwargs)

        # Convert this batch to pandas right away
        contrib_df = utils._to_pandas(contrib_chunk_hf)
        if drop_bias and "BiasTerm" in contrib_df.columns:
            contrib_df = contrib_df.drop(columns="BiasTerm")

        dfs.append(contrib_df)

        # Free H2O temporaries early
        try:
            h2o.remove(contrib_chunk_hf)
        except Exception:
            pass
        try:
            h2o.remove(chunk)
        except Exception:
            pass

        LOGGER.info(
            f"Batch {b + 1}/{batches}: {end - start} rows in {time.time() - t0:.1f}s"
        )

    # Concatenate all pandas batches once
    out_df = pd.concat(dfs, axis=0, ignore_index=True)
    LOGGER.info(f"All batches complete. Final SHAP shape: {out_df.shape}")
    return out_df


def compute_h2o_shap_contributions(
    model: H2OEstimator,
    h2o_frame: h2o.H2OFrame,
    *,
    background_data: t.Optional[h2o.H2OFrame] = None,
    drop_bias: bool = True,
    batch_rows: int = 1000,
    top_n: t.Optional[int] = None,
    bottom_n: int = 0,
    compare_abs: bool = True,
    output_format: t.Optional[str] = None,
    output_space: bool = True,
    return_features: bool = False,
) -> t.Tuple[pd.DataFrame, t.Optional[pd.DataFrame]]:
    """
    Compute SHAP/TreeSHAP contributions and optionally return input features.

    This is a wrapper around batched SHAP computation that also extracts
    the feature subset actually used by the model.

    Args:
        model: Trained H2O model.
        h2o_frame: Input frame with predictors and identifiers.
        background_data: Reference frame for SHAP baseline. Defaults to None.
        drop_bias: If True, drop the BiasTerm column. Defaults to True.
        batch_rows: Maximum number of rows per batch. Defaults to 1000.
        top_n: Return only the top N features by contribution. Defaults to None.
        bottom_n: Return only the bottom N features by contribution. Defaults to 0.
        compare_abs: Rank features by absolute contribution. Defaults to True.
        output_format: Format for output, e.g. "Compact" for XGBoost. Defaults to None.
        output_space: If True, return contributions in the model’s response
            space (e.g., probabilities). If False, keep link space. Defaults to True.
        return_features: If True, also return a DataFrame of input features
            corresponding to the contribution rows. Defaults to True.

    Returns:
        Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
            - contribs_df: SHAP contributions aligned to rows of `h2o_frame`.
            - features_df: Pandas DataFrame of input features if `return_features=True`,
              else None.

    """
    LOGGER.info("Computing SHAP contributions with batching...")

    # Select only the features the model actually uses
    used_features = get_h2o_used_features(model)
    if used_features:
        hf_subset = h2o_frame[used_features]
        bg_subset = (
            background_data[used_features] if background_data is not None else None
        )
    else:
        hf_subset = h2o_frame
        bg_subset = background_data

    # Compute contributions on the subset
    contribs_df = predict_contribs_batched(
        model,
        hf_subset,
        batch_rows=batch_rows,
        top_n=top_n,
        bottom_n=bottom_n,
        compare_abs=compare_abs,
        output_format=output_format,
        background_frame=bg_subset,
        drop_bias=drop_bias,
        output_space=output_space,
    )

    # Convert the same subset to pandas so columns line up with SHAP
    features_df: t.Optional[pd.DataFrame] = None
    if return_features:
        features_df = utils._to_pandas(hf_subset)

    LOGGER.info(
        f"Finished SHAP computation. SHAP={contribs_df.shape}"
        f"{'' if features_df is None else f', features={features_df.shape}'}"
    )
    return (contribs_df, features_df) if return_features else contribs_df


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
            out[base] = pd.concat(cols, axis=1).apply(lambda row: None, axis=1)

    return pd.DataFrame(out)


def create_color_hint_features(
    grouped_df: pd.DataFrame,
    original_dtypes: dict[str, t.Any],
) -> pd.DataFrame:
    """Build a color-hint frame for SHAP: categorical cols → string values; numeric/bool → numeric values.

    Args:
        grouped_df: Features after grouping one-hots/missing flags; rows align with SHAP rows.
        original_dtypes: Mapping {col_name: dtype-like} from the raw data (pre-imputation/encoding).
                         Values may be strings (e.g., "category", "int64"); they are normalized.

    Returns:
        DataFrame shaped like `grouped_df`. For columns considered categorical,
        the series is cast to pandas string dtype; for numeric/bool, values are kept numeric.
    """
    out = pd.DataFrame(index=grouped_df.index)

    for col in grouped_df.columns:
        dt_raw = original_dtypes.get(col, None)
        try:
            dt = pandas_dtype(dt_raw) if dt_raw is not None else None
        except Exception:
            dt = None

        is_cat = (
            dt is not None
            and (
                is_object_dtype(dt)
                or isinstance(dt, CategoricalDtype)
                or is_string_dtype(dt)
            )
            and not is_bool_dtype(dt)
        )

        # Cast categorical columns to string; keep numeric/bool as-is
        out[col] = grouped_df[col].astype("string") if is_cat else grouped_df[col]

    return out


def get_base_feature_name(col: str, group_missing_flags: bool) -> str:
    """
    Derives the base feature name used for grouping SHAP values or input features.

    Args:
        col: Column name (possibly one-hot or missing flag)
        group_missing_flags: Whether to group missing flags with base feature

    Returns:
        Base feature name for grouping
    """
    base = re.split(r"\.", col, maxsplit=1)[0]

    if group_missing_flags and base.endswith("_missing_flag"):
        return base[: -len("_missing_flag")]

    return base


def plot_grouped_shap(
    contribs_df: pd.DataFrame,
    features_df: pd.DataFrame,
    *,
    group_missing_flags: bool = False,
    original_dtypes: t.Optional[dict[str, t.Any]] = None,
    max_display: int = 20,
    mlflow_name: str = "h2o_feature_importances_by_shap_plot.png",
) -> None:
    """
    Plot grouped SHAP values as a global summary plot. One-hot encoded features are grouped under their base feature name.
    Missingness flags are optionally grouped with their base feature based on the `group_missing_flags` flag.
    A color hint matrix is built from the raw data to improve SHAP summary plot interpretability.

    Parameters:
        contribs_df: DataFrame of SHAP contributions (from H2O), including one-hot or exploded categorical features.
        features_df: Feature matrix (e.g., after imputation), matching SHAP columns.
        original_dtypes: Dictionary with dtypes from raw data (before imputation), used for inferring data types and color hints.
        group_missing_flags: Whether to group missingness flag columns (e.g., 'math_placement_missing_flag')
                             into their corresponding base feature (e.g., 'math_placement') in the SHAP plot.
    """
    # Group SHAP and features to base names
    grouped_shap = group_shap_values(
        contribs_df, group_missing_flags=group_missing_flags
    )
    grouped_feats = group_feature_values(
        features_df, group_missing_flags=group_missing_flags
    )

    # Build color hint if we have original dtypes; otherwise use grouped features directly
    # NOTE: original dtypes should be available from sklearn imputer step during training
    if original_dtypes is not None:
        color_hint = create_color_hint_features(
            grouped_df=grouped_feats, original_dtypes=original_dtypes
        )
        features_for_plot = color_hint
    else:
        features_for_plot = grouped_feats  # no color hint

    # Plot + log
    shap.summary_plot(
        grouped_shap.values,
        features=features_for_plot,
        feature_names=grouped_shap.columns,
        max_display=max_display,
        show=False,
    )
    mlflow.log_figure(plt.gcf(), mlflow_name)
