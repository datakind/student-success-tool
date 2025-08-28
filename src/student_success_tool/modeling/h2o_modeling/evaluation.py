import typing as t
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
from h2o.automl import H2OAutoML
from h2o.estimators.estimator_base import H2OEstimator

from . import training
from . import utils
from . import imputation
from . import inference

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
        thresh_df = utils._to_pandas(perf.thresholds_and_metric_scores())
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


def extract_number_of_runs_from_model_training(
    automl_experiment_id: str,
    data_runname: str = "H2O AutoML Experiment Summary and Storage",
) -> int:
    """
    Read number of runs from an H2O model. This is available from the h2o_leaderboard.csv,
    which is saved under the "H2O AutoML Experiment Summary and Storage" run which is available
    in an experiment. The leaderboard contains information on every run that was created during
    H2O training. We only log the top 50 models, so it's typically hundreds of models. Each row
    is a separate run in h2o_leaderboard.csv so the length of the dataframe is the number of models.

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
        run_id=data_run_id, artifact_path="leaderboard", dst_path=input_temp_dir
    )
    df_leaderboard = pd.read_csv(os.path.join(input_data_path, "h2o_leaderboard.csv"))
    # Delete the temp data
    shutil.rmtree(input_temp_dir)

    return int(df_leaderboard.shape[0])


############
## PLOTS! ##
############


def create_and_log_h2o_model_comparison(
    aml: H2OAutoML,
    artifact_path: str = "model_comparison.png",
) -> pd.DataFrame:
    """
    Plots best (lowest) logloss per framework using AutoML leaderboard metrics,
    logs the figure to MLflow, and returns the compact DataFrame used for plotting.
    """
    included_frameworks = set(training.VALID_H2O_FRAMEWORKS)

    lb = utils._to_pandas(aml.leaderboard)

    # Ensure there's a 'framework' column
    if "algo" in lb.columns:
        df = lb.rename(columns={"algo": "framework"})
    else:
        df = lb.copy()
        # infer framework by splitting model_id at '_' and taking first token
        df["framework"] = df["model_id"].str.split("_").str[0]

    # Keep only frameworks we trained with
    df = df.loc[
        df["framework"].isin(included_frameworks), ["framework", "logloss"]
    ].dropna()

    # Best (lowest) per family, sorted low→high
    best = (
        df.sort_values("logloss", ascending=True)
        .drop_duplicates(subset=["framework"], keep="first")
        .sort_values("logloss", ascending=True)
        .reset_index(drop=True)
    )

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(best["framework"], best["logloss"])

    if len(bars):
        bars[0].set_alpha(1.0)
        for b in bars[1:]:
            b.set_alpha(0.5)
        for i, b in enumerate(bars):
            ax.text(
                b.get_width() * 0.98,
                b.get_y() + b.get_height() / 2,
                f"{best['logloss'].iloc[i]:.4f}",
                va="center",
                ha="right",
            )

    ax.set_xlabel("log_loss")
    ax.set_title("log_loss by Model Type (lowest to highest)")
    ax.set_xlim(left=0)
    ax.invert_yaxis()
    plt.tight_layout()

    if mlflow.active_run():
        mlflow.log_figure(fig, artifact_path)

    plt.close(fig)
    return best


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


def log_roc_table(
    institution_id: str,
    *,
    automl_run_id: str,
    catalog: str = "staging_sst_01",
    target_col: str = "target",
    modeling_dataset_name: str = "modeling_dataset",
    split_col: t.Optional[str] = None,
) -> None:
    """
    Computes and saves an ROC curve table (FPR, TPR, threshold, etc.) for a given H2O model run
    by reloading the test dataset and the trained model.

    Args:
        institution_id (str): Institution ID prefix for table name.
        automl_experiment_id (str): MLflow run ID of the trained model.
        experiment_id
        catalog (str): Destination catalog/schema for the ROC curve table.
    """
    try:
        from databricks.connect import DatabricksSession

        spark = DatabricksSession.builder.getOrCreate()
    except Exception:
        print("⚠️ Databricks Connect failed. Falling back to local Spark.")
        from pyspark.sql import SparkSession

        spark = (
            SparkSession.builder.master("local[*]").appName("Fallback").getOrCreate()
        )

    split_col = split_col or "split"

    table_path = f"{catalog}.{institution_id}_silver.training_{automl_run_id}_roc_curve"

    try:
        df = spark.read.table(modeling_dataset_name).toPandas()
        test_df = df[df[split_col] == "test"].copy()

        # Load and transform using sklearn imputer
        test_df = imputation.SklearnImputerWrapper.load_and_transform(
            test_df,
            run_id=automl_run_id,
        )

        # Load model + features
        model = utils.load_h2o_model(automl_run_id)
        feature_names: t.List[str] = inference.get_h2o_used_features(model)

        # Prepare inputs for ROC
        y_true = test_df[target_col].values
        X_test = test_df[feature_names]
        _, y_scores = inference.predict_h2o(
            X_test,
            model=model,
        )

        # Calculate ROC table manually and plot all thresholds.
        # Down the line, we might want to specify a threshold to reduce plot density
        thresholds = np.sort(np.unique(y_scores))[::-1]
        rounded_thresholds = sorted(
            set([round(t, 4) for t in thresholds]), reverse=True
        )

        P, N = np.sum(y_true == 1), np.sum(y_true == 0)

        rows = []
        for thresh in rounded_thresholds:
            y_pred = (y_scores >= thresh).astype(int)
            TP = np.sum((y_pred == 1) & (y_true == 1))
            FP = np.sum((y_pred == 1) & (y_true == 0))
            TN = np.sum((y_pred == 0) & (y_true == 0))
            FN = np.sum((y_pred == 0) & (y_true == 1))
            TPR = TP / P if P else 0
            FPR = FP / N if N else 0
            rows.append(
                {
                    "threshold": round(thresh, 4),
                    "true_positive_rate": round(TPR, 4),
                    "false_positive_rate": round(FPR, 4),
                    "true_positive": int(TP),
                    "false_positives": int(FP),
                    "true_negatives": int(TN),
                    "false_negatives": int(FN),
                }
            )

        roc_df = pd.DataFrame(rows)
        spark_df = spark.createDataFrame(roc_df)
        spark_df.write.mode("overwrite").saveAsTable(table_path)
        logging.info(
            "ROC table written to table '%s' for run_id=%s", table_path, automl_run_id
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to log ROC table for run {automl_run_id}: {e}"
        ) from e
