import logging
import os
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import h2o
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    RocCurveDisplay,
    precision_recall_curve,
    PrecisionRecallDisplay,
    auc
)


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


def compute_lift(y_true, y_scores, n_bins=10):
    df = pd.DataFrame({'y_true': y_true, 'y_score': y_scores})
    df = df.sort_values('y_score', ascending=False)
    df['bin'] = pd.qcut(df['y_score'], q=n_bins, duplicates='drop')
    lift = df.groupby('bin')['y_true'].mean() / df['y_true'].mean()
    return lift

############
## PLOTS! ##
############

def create_confusion_matrix_plot(y_true, y_pred) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax)
    plt.close(fig)
    return fig


def create_roc_curve_plot(y_true, y_proba) -> plt.Figure:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label="ROC Curve")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    plt.close(fig)
    return fig


def create_precision_recall_curve_plot(y_true, y_proba) -> plt.Figure:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    fig, ax = plt.subplots()
    ax.plot(recall, precision, label="Precision-Recall Curve")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    plt.close(fig)
    return fig


def create_lift_curve_plot(y_true, y_proba, n_bins=10) -> plt.Figure:
    df = pd.DataFrame({"y_true": y_true, "y_proba": y_proba})
    df["bin"] = pd.qcut(df["y_proba"], q=n_bins, duplicates='drop')
    lift = df.groupby("bin")["y_true"].mean()
    baseline = df["y_true"].mean()

    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(lift) + 1), lift / baseline, marker="o", label="Lift")
    ax.axhline(y=1, color="gray", linestyle="--")
    ax.set_title("Lift Curve")
    ax.set_xlabel("Bin (by predicted probability)")
    ax.set_ylabel("Lift over baseline")
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
        "confusion_matrix": create_confusion_matrix_plot,
        "roc_curve": create_roc_curve_plot,
        "precision_recall": create_precision_recall_curve_plot,
        "lift_curve": create_lift_curve_plot,
    }

    for name, plot_fn in plot_fns.items():
        fig = plot_fn(y_true, y_proba if "proba" in plot_fn.__name__ else y_pred)
        mlflow.log_figure(fig, f"{prefix}_{name}.png")