import logging
from collections.abc import Callable

import shutil
import uuid

import os
import mlflow

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
)
from sklearn.calibration import calibration_curve


import h2o
from h2o.estimators.estimator_base import H2OEstimator

LOGGER = logging.getLogger(__name__)


# --- Metric evaluation ---
def get_metrics_near_threshold_all_splits(
    model: H2OEstimator,
    train: h2o.H2OFrame,
    valid: h2o.H2OFrame,
    test: h2o.H2OFrame,
    threshold: float = 0.5,
) -> dict[str, float | str]:
    def _metrics(perf, label):
        thresh_df = perf.thresholds_and_metric_scores().as_data_frame()
        closest = thresh_df.iloc[
            (thresh_df["threshold"] - threshold).abs().argsort()[:1]
        ]
        return {
            f"{label}_threshold": closest["threshold"].values[0],
            f"{label}_precision": closest["precision"].values[0],
            f"{label}_recall": closest["recall"].values[0],
            f"{label}_accuracy": closest["accuracy"].values[0],
            f"{label}_f1": closest["f1"].values[0],
            f"{label}_roc_auc": perf.auc(),
            f"{label}_log_loss": perf.logloss(),
        }

    return {
        "model_id": model.model_id,
        **_metrics(model.model_performance(train), "train"),
        **_metrics(model.model_performance(valid), "validate"),
        **_metrics(model.model_performance(test), "test"),
    }


def generate_all_classification_plots(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, prefix: str = "test"
) -> None:
    """
    Generates and logs classification plots to MLflow as figures.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted class labels
        y_proba: Predicted probabilities for the positive class
        prefix: Prefix for plot file names (e.g., "train", "test", "val")
    """

    plot_fns: dict[
        str, tuple[Callable[[np.ndarray, np.ndarray], plt.Figure], np.ndarray]
    ] = {
        "confusion_matrix": (create_confusion_matrix_plot, y_pred),
        "roc_curve": (create_roc_curve_plot, y_proba),
        "precision_recall": (create_precision_recall_curve_plot, y_proba),
        "calibration_curve": (create_calibration_curve_plot, y_proba),
    }

    for name, (plot_fn, values) in plot_fns.items():
        fig = plot_fn(y_true, values)
        mlflow.log_figure(fig, f"{prefix}_{name}.png")


def extract_training_data_from_model(
    automl_experiment_id: str,
    data_runname: str = "H2O AutoML Experiment Summary and Storage",
) -> pd.DataFrame:
    """
    Read training data from a model into a pandas DataFrame. This allows us to run more
    evaluations of the model, ensuring that we are using the same train/test/validation split

    Args:
        automl_experiment_id: Experiment ID of the AutoML experiment
        data_runname: The runName tag designating where there training data is stored

    Returns:
        The data used for training a model, with train/test/validation flags
    """
    run_df = mlflow.search_runs(
        experiment_ids=[automl_experiment_id], output_format="pandas"
    )
    assert isinstance(run_df, pd.DataFrame)  # type guard
    data_run_id = run_df[run_df["tags.mlflow.runName"] == data_runname]["run_id"].item()

    # Create temp directory to download input data from MLflow
    input_temp_dir = os.path.join(
        os.environ["SPARK_LOCAL_DIRS"], "tmp", str(uuid.uuid4())[:8]
    )
    os.makedirs(input_temp_dir)

    # Download the artifact and read it into a pandas DataFrame
    input_data_path = mlflow.artifacts.download_artifacts(
        run_id=data_run_id, artifact_path="inputs", dst_path=input_temp_dir
    )
    df_loaded = pd.read_parquet(os.path.join(input_data_path, "full_dataset.parquet"))
    # Delete the temp data
    shutil.rmtree(input_temp_dir)

    return df_loaded


############
## PLOTS! ##
############


def create_confusion_matrix_plot(y_true: np.ndarray, y_pred: np.ndarray) -> plt.Figure:
    # Normalize confusion matrix by true labels
    cm = confusion_matrix(y_true, y_pred, normalize="true")

    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)

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


def create_roc_curve_plot(y_true: np.ndarray, y_proba: np.ndarray) -> plt.Figure:
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


def create_precision_recall_curve_plot(
    y_true: np.ndarray, y_proba: np.ndarray
) -> plt.Figure:
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


def create_calibration_curve_plot(
    y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10
) -> plt.Figure:
    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(
        y_true, y_proba, n_bins=n_bins, strategy="quantile"
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
