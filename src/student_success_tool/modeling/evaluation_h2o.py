import logging
import typing as t
import os
import re
import mlflow

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import (
    is_object_dtype,
    is_string_dtype,
    is_bool_dtype,
)
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    RocCurveDisplay,
    precision_recall_curve,
    PrecisionRecallDisplay,
    auc,
    roc_auc_score,
    average_precision_score,
)
from sklearn.calibration import calibration_curve


import h2o
from h2o.automl import H2OAutoML
import shap

LOGGER = logging.getLogger(__name__)


# --- Metric evaluation ---
def get_metrics_near_threshold_all_splits(model, train, valid, test, threshold=0.5):
    def _metrics(perf, label):
        thresh_df = perf.thresholds_and_metric_scores().as_data_frame()
        closest = thresh_df.iloc[(thresh_df['threshold'] - threshold).abs().argsort()[:1]]
        return {
            f"{label}_threshold": closest['threshold'].values[0],
            f"{label}_precision": closest['precision'].values[0],
            f"{label}_recall": closest['recall'].values[0],
            f"{label}_accuracy": closest['accuracy'].values[0],
            f"{label}_f1": closest['f1'].values[0],
            f"{label}_roc_auc": perf.auc(),
            f"{label}_log_loss": perf.logloss()
        }

    return {
        "model_id": model.model_id,
        **_metrics(model.model_performance(train), "train"),
        **_metrics(model.model_performance(valid), "validate"),
        **_metrics(model.model_performance(test), "test"),
    }


############
## PLOTS! ##
############

def create_confusion_matrix_plot(y_true, y_pred) -> plt.Figure:
    # Normalize confusion matrix by true labels
    cm = confusion_matrix(y_true, y_pred, normalize='true')

    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap='Blues', colorbar=False)

    # Remove default annotations
    for txt in ax.texts:
        txt.set_visible(False)

    # Dynamic contrast-aware text overlay
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            # Use white text on dark blue, black on light blue
            text_color = "black" if value < 0.5 else "white"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color)

    ax.set_title("Normalized Confusion Matrix")
    plt.tight_layout()
    plt.close(fig)
    return fig


def create_roc_curve_plot(y_true, y_proba) -> plt.Figure:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.2f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    plt.close(fig)
    return fig


def create_precision_recall_curve_plot(y_true, y_proba) -> plt.Figure:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap_score = average_precision_score(y_true, y_proba)

    fig, ax = plt.subplots()
    ax.plot(recall, precision, label=f"Precision-Recall (AP = {ap_score:.2f})")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    plt.close(fig)
    return fig



def create_calibration_curve_plot(y_true, y_proba, n_bins=10) -> plt.Figure:
    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(
        y_true, y_proba, n_bins=n_bins, strategy='quantile'
    )

    # Create the plot
    fig, ax = plt.subplots()
    ax.plot(prob_pred, prob_true, marker="o", label="Model Calibration")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")

    # Labels and legend
    ax.set_title("Calibration Curve")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    plt.close(fig)
    return fig


def generate_all_classification_plots(y_true, y_pred, y_proba, prefix="test"):
    """
    Generates and logs classification plots to MLflow as figures.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted class labels
        y_proba: Predicted probabilities for the positive class
        prefix: Prefix for plot file names (e.g., "train", "test", "val")
    """
    plot_fns = {
        "confusion_matrix": (create_confusion_matrix_plot, y_pred),
        "roc_curve": (create_roc_curve_plot, y_proba),
        "precision_recall": (create_precision_recall_curve_plot, y_proba),
        "calibration_curve": (create_calibration_curve_plot, y_proba),
    }

    for name, (plot_fn, values) in plot_fns.items():
        fig = plot_fn(y_true, values)
        mlflow.log_figure(fig, f"{prefix}_{name}.png")


def get_h2o_used_features(model):
    """
    Extracts the actual feature names used by the H2O model (excluding dropped/constant columns).
    """
    # The last name is usually the response/target variable
    feature_names = model._model_json['output']['names'][:-1]
    return feature_names


def compute_h2o_shap_contributions(
    model: H2OModel,
    h2o_frame: H2OFrame,
    background_data: t.Optional[H2OFrame] = None,
    drop_bias: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes SHAP-like contribution values from an H2O model.

    Args:
        model: Trained H2O model
        h2o_frame: H2OFrame for which to compute contributions
        background_data: Optional H2OFrame to use as the background reference for SHAP values
        drop_bias: Whether to exclude the 'BiasTerm' column

    Returns:
        contribs_df: SHAP contributions aligned with input features
        input_df: Input feature values used
    """
    used_features = get_h2o_used_features(model)
    hf_subset = h2o_frame[used_features]

    if background_data is not None:
        background_data = background_data[used_features]
        contribs_hf = model.predict_contributions(hf_subset, background_data=background_data)
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


def create_color_hint_features(original_df: pd.DataFrame, grouped_df: pd.DataFrame) -> pd.DataFrame:
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
                (is_object_dtype(dtype) or isinstance(dtype, pd.CategoricalDtype) or is_string_dtype(dtype))
                and not is_bool_dtype(dtype)
            )
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


def plot_grouped_shap(contribs_df: pd.DataFrame, input_df: pd.DataFrame, original_df: pd.DataFrame):
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
        grouped_shap.values,
        features=color_hint,
        feature_names=grouped_shap.columns
    )
