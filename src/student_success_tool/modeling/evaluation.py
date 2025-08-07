import logging
import os
import shutil
import typing as t
import uuid
from collections.abc import Sequence
from typing import Optional, List
import matplotlib.colors as mcolors
import matplotlib.figure
import matplotlib.pyplot as plt
import mlflow
import mlflow.artifacts
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import sklearn.inspection
import sklearn.metrics
import sklearn.utils
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import MinMaxScaler
from statsmodels.nonparametric.smoothers_lowess import lowess

try:
    from IPython.display import display
except ImportError:
    display = t.cast(t.Callable[..., t.Any], print)

LOGGER = logging.getLogger(__name__)

# TODO: eventually we should use the custom_style.mplstyle colors, but currently
# the color palette does not have distinct enough colors for calibration by group
# where there are a lot of subgroups
PALETTE = sns.color_palette("Paired")

PosLabelType = t.Optional[int | float | bool | str]


def extract_training_data_from_model(
    automl_experiment_id: str, data_runname: str = "Training Data Storage and Analysis"
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
        run_id=data_run_id, artifact_path="data", dst_path=input_temp_dir
    )
    df_loaded = pd.read_parquet(os.path.join(input_data_path, "training_data"))
    # Delete the temp data
    shutil.rmtree(input_temp_dir)

    return df_loaded


def evaluate_performance(
    df_pred: pd.DataFrame,
    *,
    pos_label: PosLabelType,
    split_col: str = "split",
    target_col: str = "target",
    pred_col: str = "pred",
    pred_prob_col: str = "pred_prob",
) -> None:
    """
    Evaluates and logs model performance for each data split. Generates
    histogram and calibration plots.

    Args:
        df_pred: DataFrame containing prediction results with a column for splits.
        split_col: Column name indicating split column ("train", "test", or "val").
        target_col: Column name for the target variable.
        pred_prob_col: Column name for predicted probabilities.
        pos_label: Positive class label.
    """
    calibration_dir = "calibration"
    preds_dir = "preds"
    metrics_dir = "metrics"
    metrics_records = []

    for split_name, split_data in df_pred.groupby(split_col):
        LOGGER.info("Evaluating model performance for '%s' split", split_name)
        split_data.to_csv(f"/tmp/{split_name}_preds.csv", header=True, index=False)
        mlflow.log_artifact(
            local_path=f"/tmp/{split_name}_preds.csv",
            artifact_path=preds_dir,
        )
        hist_fig, cal_fig = create_evaluation_plots(
            split_data,
            pred_prob_col,
            target_col,
            pos_label,
            split_name,
        )
        mlflow.log_figure(hist_fig, os.path.join(preds_dir, f"{split_name}_hist.png"))
        mlflow.log_figure(
            cal_fig, os.path.join(calibration_dir, f"{split_name}_calibration.png")
        )
        # Closes all matplotlib figures in console to free memory
        plt.close("all")

        # Compute performance metrics by split (returns a dict)
        perf_metrics_raw = compute_classification_perf_metrics(
            targets=split_data[target_col],
            preds=split_data[pred_col],
            pred_probs=split_data[pred_prob_col],
            pos_label=pos_label,
        )

        perf_metrics = format_perf_metrics(perf_metrics_raw)
        perf_split_col = f"Dataset {split_col.capitalize()}"
        split_map = {"test": "Test", "train": "Training", "validate": "Validation"}
        perf_metrics[perf_split_col] = split_map.get(split_name)
        metrics_records.append(perf_metrics)

    # Convert to DataFrame for display or saving
    metrics_df = pd.DataFrame(metrics_records).set_index(perf_split_col)
    metrics_df.to_csv("/tmp/performance_across_splits.csv")
    mlflow.log_artifact("/tmp/performance_across_splits.csv", artifact_path=metrics_dir)
    LOGGER.info("Creating summary of performance metrics across splits")


def format_perf_metrics(perf_metrics_raw: dict) -> dict:
    """
    Formats performance metrics from raw metrics. This is for a model cards
    table that presents performance by splits.

    Args:
        perf_metrics_raw: Dictionary containing raw performance metrics.

    Returns:
        Dictionary summarizing the formatted performance metrics.
    """
    return {
        "Number of Samples": int(perf_metrics_raw["num_samples"]),
        "Number of Positive Samples": int(perf_metrics_raw["num_positives"]),
        "Actual Target Prevalence": round(
            float(perf_metrics_raw["true_positive_prevalence"]), 2
        ),
        "Predicted Target Prevalence": round(
            float(perf_metrics_raw["pred_positive_prevalence"]), 2
        ),
        "Accuracy": round(float(perf_metrics_raw["accuracy"]), 2),
        "Precision": round(float(perf_metrics_raw["precision"]), 2),
        "Recall": round(float(perf_metrics_raw["recall"]), 2),
        "F1 Score": round(float(perf_metrics_raw["f1_score"]), 2),
        "Log Loss": round(float(perf_metrics_raw["log_loss"]), 2),
    }


def get_top_runs(
    experiment_id: str,
    optimization_metrics: t.Union[str, list[str]],
    topn_runs_included: int = 1,
) -> dict[str, str]:
    """
    Retrieve top runs from an MLflow experiment using a balanced score over multiple metrics.

    Parameters:
        experiment_id: MLflow experiment ID.
        optimization_metrics: Single metric or list of metrics to sort by.
        topn_runs_included: Number of top runs to return.

    Returns:
        A dictionary mapping run_name to run_id.
    """
    if isinstance(optimization_metrics, str):
        optimization_metrics = [optimization_metrics]

    # Prepare column names and directions
    sort_by = [f"metrics.{m}" for m in optimization_metrics]
    directions = infer_directions(optimization_metrics)

    # Fetch all runs
    runs = mlflow.search_runs(experiment_ids=[experiment_id], output_format="pandas")
    assert isinstance(runs, pd.DataFrame)

    # Clean and filter
    valid_runs = runs.dropna(subset=sort_by).copy()
    for col in sort_by:
        valid_runs[col] = valid_runs[col].astype(float)

    # Score and sort
    valid_runs["balanced_score"] = compute_balanced_score(
        valid_runs, sort_by, directions
    )
    sorted_runs = valid_runs.sort_values("balanced_score")

    # Display viz of top runs
    display_cols = ["tags.mlflow.runName", "run_id"] + sort_by + ["balanced_score"]

    print("Inferred directions:")
    for metric, direction in zip(optimization_metrics, directions):
        print(f"  • {metric}: {'minimize' if direction == 'asc' else 'maximize'}")

    print(f"\n Top {topn_runs_included} runs (by balanced score):")
    display(sorted_runs[display_cols].head(topn_runs_included))

    # Return as {run_name: run_id}
    top_runs = sorted_runs.head(topn_runs_included)
    return {
        row.get("tags.mlflow.runName", f"Unnamed_{row['run_id']}"): row["run_id"]
        for _, row in top_runs.iterrows()
    }


def infer_directions(metrics: list[str]) -> list[str]:
    """
    When getting top runs, we can infer optimization direction
    for each metric: 'asc' (minimize) or 'desc' (maximize).

    Parameters:
        metrics: List of metric names.

    Returns:
        List of directions, one for each metric.
    """
    minimize_keywords = (
        "loss",
        "error",
        "log_loss",
        "mae",
        "mse",
        "bias_score_mean",
        "bias_score_max",
    )
    return [
        "asc" if any(metric.lower().endswith(k) for k in minimize_keywords) else "desc"
        for metric in metrics
    ]


def compute_balanced_score(
    df: pd.DataFrame, metrics: list[str], directions: list[str]
) -> pd.Series:
    """
    When sorting through top runs, we need to average normalize our metrics in order to
    properly find the best overall runs across all given metrics. We balance ALL metrics
    equally. We then compute a balanced average score (lower is better).

    TODO: Provide a priority list of metrics: higher recall, then ROC AUC, then log loss,
    for example.

    Parameters:
        df: Pandas DataFrame containing metrics
        metrics: List of metric names
        directions: List of directions for each metric

    Returns:
        Balanced score for each row in the DataFrame
    """
    scaler = MinMaxScaler()
    data = df[metrics].copy()
    normalized = scaler.fit_transform(data)

    # Flip maximization metrics so that lower = better for everything
    for i, direction in enumerate(directions):
        if direction == "desc":
            normalized[:, i] = 1 - normalized[:, i]

    return pd.Series(normalized.mean(axis=1), index=df.index)


def compute_classification_perf_metrics(
    targets: pd.Series,
    preds: pd.Series,
    pred_probs: pd.Series,
    *,
    pos_label: PosLabelType,
    sample_weights: t.Optional[pd.Series] = None,
) -> dict[str, object]:
    """
    Compute a variety of useful metrics for evaluating the performance of a classifier.

    Args:
        targets: True target values that classifier model was trained to predict.
        preds: Binary predictions output by classifier model.
        pred_probs: Prediction probabilities output by classifier model,
            for the positive class only.
        pos_label: Value of the positive class (aka "label").
        sample_weights: Sample weights for the examples corresponding to targets/preds,
            if classifier model was trained on weighted samples.
    """
    assert isinstance(pos_label, (int, float, bool, str))  # type guard
    precision, recall, f1_score, _ = sklearn.metrics.precision_recall_fscore_support(
        targets,
        preds,
        beta=1,
        average="binary",
        pos_label=pos_label,  # type: ignore
        sample_weight=sample_weights,
        zero_division=np.nan,  # type: ignore
    )
    result = {
        "num_samples": len(targets),
        "num_positives": targets.eq(pos_label).sum(),
        "true_positive_prevalence": targets.eq(pos_label).mean(),
        "pred_positive_prevalence": preds.eq(pos_label).mean(),
        "accuracy": sklearn.metrics.accuracy_score(
            targets, preds, sample_weight=sample_weights
        ),
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "log_loss": sklearn.metrics.log_loss(
            targets, pred_probs, sample_weight=sample_weights, labels=[False, True]
        ),
        # TODO: add this back in, but handle errors for small N case
        # "roc_auc": sklearn.metrics.roc_auc_score(
        #     targets, pred_probs, sample_weight=sample_weights, labels=[False, True]
        # ),
    }
    return result


def get_sensitivity_of_top_q_pctl_thresh(
    y_true: pd.Series | Sequence,
    risk_score: pd.Series | Sequence,
    q: float,
    pos_label: PosLabelType,
) -> float:
    """
    Report sensitivity (AKA recall score) using some percentile threshold.

    Calculation:
        number of true positivies / (number of true positives + number of false negatives)
        OR
        number of true positives / total number of actual true class

    Args:
        y_true: true outcome class, of length n_samples
        risk_score: predicted risk scores, of length n_samples
        q: probability for the quantiles to compute
        pos_label: label identifying the positive class in y_true
    """
    if not isinstance(y_true, pd.Series):
        y_true = pd.Series(y_true)
    if not isinstance(risk_score, pd.Series):
        risk_score = pd.Series(risk_score)

    prob_thresh = np.quantile(risk_score, q)
    high_risk = risk_score >= prob_thresh

    # convert actual outcome to booleans to match high_risk array
    y_true = y_true.apply(lambda x: True if x == pos_label else False)

    result = sklearn.metrics.recall_score(y_true, high_risk)
    assert isinstance(result, float)
    return result


def compare_trained_models(
    automl_experiment_id: str, automl_metric: str
) -> tuple[pd.DataFrame, str]:
    """
    Retrieve, aggregate and sort performance data for models trained in a specified AutoML experiment.
    The validation dataset is used to tune hyperparameters. Metrics on the validation dataset are used to rank models, so we also use this metric to compare across models.

    Args:
        automl_experiment_id: Experiment ID of the AutoML experiment
        automl_metric: Chosen AutoML optimization metric

    Returns:
        DataFrame containing model types and highest scores for the given metric.
    """
    runs = mlflow.search_runs(
        experiment_ids=[automl_experiment_id], output_format="pandas"
    )
    assert isinstance(runs, pd.DataFrame)  # type guard
    metric = str.lower(automl_metric)
    metric_score_column = (
        f"metrics.val_{metric}"
        if metric == "log_loss"
        else f"metrics.val_{metric}_score"
    )

    runs = runs[["tags.model_type", metric_score_column]].dropna()

    ascending = metric == "log_loss"
    df_sorted = (
        runs.groupby("tags.model_type", as_index=False)
        .agg({metric_score_column: ("min" if ascending else "max")})
        .sort_values(by=metric_score_column, ascending=ascending)
    )
    return df_sorted, metric_score_column


def compute_feature_permutation_importance(
    model: sklearn.base.BaseEstimator,
    features: pd.DataFrame,
    target: pd.Series,
    *,
    n_repeats: int = 5,
    scoring: t.Optional[str | t.Callable] = None,
    sample_weight: t.Optional[np.ndarray] = None,
    random_state: t.Optional[int] = None,
) -> sklearn.utils.Bunch:
    """
    Compute model features' permutation importance.

    Args:
        model
        features
        target
        n_repeats
        scoring
        sample_weight
        random_state

    References:
        - https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html
        - https://scikit-learn.org/stable/modules/model_evaluation.html#callable-scorers

    See Also:
        - :func:`plot_feature_permutation_importances()`
    """
    result = sklearn.inspection.permutation_importance(
        model,
        features,
        target,
        scoring=scoring,
        n_repeats=n_repeats,
        sample_weight=sample_weight,
        random_state=random_state,
    )
    assert isinstance(result, sklearn.utils.Bunch)  # type guard
    return result


def log_confusion_matrix(
    institution_id: str,
    *,
    automl_run_id: str,
    catalog: str = "staging_sst_01",
) -> None:
    """
    Register an mlflow model according to one of their various recommended approaches.

    Args:
        institution_id
        automl_run_id
        catalog
    """
    try:
        from databricks.connect import DatabricksSession

        spark = DatabricksSession.builder.getOrCreate()
    except Exception:
        print("Databricks Connect failed. Falling back to local Spark.")
        from pyspark.sql import SparkSession

        spark = (
            SparkSession.builder.master("local[*]").appName("Fallback").getOrCreate()
        )

    confusion_matrix_table_path = (
        f"{catalog}.{institution_id}_silver.training_{automl_run_id}_confusion_matrix"
    )

    def safe_div(numerator, denominator):
        return numerator / denominator if denominator else 0.0

    if mlflow.active_run() is None:
        raise RuntimeError(
            "No active MLflow run. Please call this within an MLflow run context."
        )

    try:
        run = mlflow.get_run(automl_run_id)
        required_metrics = [
            "test_true_positives",
            "test_true_negatives",
            "test_false_positives",
            "test_false_negatives",
        ]
        metrics = {m: run.data.metrics.get(m) for m in required_metrics}

        if any(v is None for v in metrics.values()):
            raise ValueError(
                f"Missing one or more required metrics in run {automl_run_id}: {metrics}"
            )

        tp, tn, fp, fn = (
            metrics["test_true_positives"],
            metrics["test_true_negatives"],
            metrics["test_false_positives"],
            metrics["test_false_negatives"],
        )

        tn_percentage = safe_div(tn, tn + fp)
        tp_percentage = safe_div(tp, tp + fn)
        fp_percentage = safe_div(fp, fp + tn)
        fn_percentage = safe_div(fn, fn + tp)

        confusion_matrix_table = pd.DataFrame(
            {
                "true_positive": [tp_percentage],
                "false_positive": [fp_percentage],
                "true_negative": [tn_percentage],
                "false_negative": [fn_percentage],
            }
        )

        confusion_matrix_table_spark = spark.createDataFrame(confusion_matrix_table)
        confusion_matrix_table_spark.write.mode("overwrite").saveAsTable(
            confusion_matrix_table_path
        )
        LOGGER.info(
            "Confusion matrix written to table '%s' for run_id=%s",
            confusion_matrix_table_path,
            automl_run_id,
        )

    except mlflow.exceptions.MlflowException as e:
        raise RuntimeError(f"MLflow error while retrieving run {automl_run_id}: {e}")
    except Exception:
        LOGGER.exception("Failed to compute or store confusion matrix.")
        raise


def log_roc_table(
    institution_id: str,
    *,
    automl_run_id: str,
    catalog: str = "staging_sst_01",
    target_col: str = "target",
    modeling_dataset_name: str = "modeling_dataset",
    split_col: Optional[str] = None,
) -> None:
    """
    Computes and saves an ROC curve table (FPR, TPR, threshold, etc.) for a given MLflow run
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

        # Load model + features
        model = mlflow.sklearn.load_model(f"runs:/{automl_run_id}/model")
        feature_names: List[str] = model.named_steps["column_selector"].get_params()[
            "cols"
        ]

        # Prepare inputs for ROC
        y_true = test_df[target_col].values
        X_test = test_df[feature_names]
        y_scores = model.predict_proba(X_test)[:, 1]  # probabilities for class 1

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


############
## PLOTS! ##
############


def create_evaluation_plots(
    data: pd.DataFrame,
    risk_score_col: str,
    y_true_col: str,
    pos_label: PosLabelType,
    split_type: str,
) -> tuple[matplotlib.figure.Figure, ...]:
    """
    Create plots to evaluate a model overall - risk score histogram
    and calibration curve at low alert rates

    Args:
        data: containing predicted and actual outcome data
        risk_score_col: column name containing data of predicted risk scores
        y_true_col: column name containing data of actual outcome classes
        pos_label: label identifying the positive class in y_true
        split_type: type of data being plotted for labeling plots - train, test, or validation

    Returns:
        risk score histogram and calibration curve at low alert rates figures
    """
    title_suffix = f"{split_type} data - Overall"
    hist_fig = plot_support_score_histogram(data[risk_score_col], title_suffix)
    cal_fig = plot_calibration_curve(
        data[y_true_col], data[risk_score_col], "Overall", title_suffix, pos_label
    )
    return hist_fig, cal_fig


def create_evaluation_plots_by_subgroup(
    data: pd.DataFrame,
    risk_score_col: str,
    y_true_col: str,
    pos_label: PosLabelType,
    group_col: str,
    split_type: str,
) -> matplotlib.figure.Figure:
    """
    Create calibration curve plots to evaluate a model by group

    Args:
        data: containing predicted and actual outcome data, as well as group label
        risk_score_col: column name containing data of predicted risk scores
        y_true_col: column name containing data of actual outcome classes
        pos_label: label identifying the positive class in y_true
        group_col: column name containing data of subgroup labels
        split_type: type of data being plotted for labeling plots - train, test, or validation

    Returns:
        calibration curve figures by group
    """
    title_suffix = f"{split_type} data - {group_col}"

    grouped_data = data.groupby(group_col).agg(list)

    ys = grouped_data[y_true_col]
    scores = grouped_data[risk_score_col]

    subgroups = grouped_data.index
    n_counts = [len(subgroup_scores) for subgroup_scores in scores]
    names = [
        subgroup_name + f" (N={n})" for subgroup_name, n in zip(subgroups, n_counts)
    ]

    if len(subgroups) > 4:
        lowess_frac = 0.7
    else:
        lowess_frac = 0.6

    cal_subgroup_plot = plot_calibration_curve(
        ys, scores, names, title_suffix, pos_label, lowess_frac=lowess_frac
    )

    return cal_subgroup_plot


def plot_trained_models_comparison(
    automl_experiment_id: str, automl_metric: str
) -> matplotlib.figure.Figure:
    """
    Create a plot to evaluate all the models trained by AutoML.

    Args:
        automl_experiment_id: Experiment ID of the AutoML experiment
        automl_metric: Chosen AutoML optimization metric

    Returns:
        bar chart of model performance on test data by optimization metric.
    """
    df_sorted, metric_score_column = compare_trained_models(
        automl_experiment_id, automl_metric
    )

    df_sorted = df_sorted.sort_values(
        by=metric_score_column,
        ascending=True if not automl_metric == "log_loss" else False,
    )
    fig, ax = plt.subplots()

    ax.set_xlim(
        0,
        (
            df_sorted[metric_score_column].max() + 0.2
            if automl_metric == "log_loss"
            else 1
        ),
    )  # setting buffer after max log_loss to 0.2, for better viz

    colors = ["#A9A9A9"] * len(df_sorted)
    colors[-1] = "#5bc0de"  # Bright blue color for the best model

    bars = ax.barh(
        df_sorted["tags.model_type"], df_sorted[metric_score_column], color=colors
    )

    for bar in bars:
        ax.text(
            bar.get_width()
            - (
                max(df_sorted[metric_score_column]) * 0.005
            ),  # Position the text inside the bar
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.4f}",
            va="center",
            ha="right",
            fontweight="bold",
            color="white",
            fontsize=12,
        )

    sort_order = (
        "Lowest to Highest" if automl_metric == "log_loss" else "Highest to Lowest"
    )
    automl_metric = (
        automl_metric
        if automl_metric == "log_loss"
        else f"{automl_metric.capitalize()} Score"
    )

    ax.set(
        title=f"{automl_metric} by Model Type {sort_order}",
        facecolor="none",
        frame_on=False,
    )
    ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=True)
    ax.tick_params(axis="x", colors="lightgrey", which="both")  # Color of ticks
    ax.xaxis.grid(True, color="lightgrey", linestyle="--", linewidth=0.5)
    fig.tight_layout()

    mlflow.log_figure(fig, "model_comparison.png")
    plt.close()

    return fig


def plot_calibration_curve(
    y_true: pd.Series,
    risk_score: pd.Series,
    keys: str | list[str],
    title_suffix: str,
    pos_label: PosLabelType,
    lowess_frac: t.Optional[float] = None,
) -> matplotlib.figure.Figure:
    """
    Plot calibration curve.

    Args:
        y_true (array-like of shape (n_samples,) or (n_groups,)): overall or group-level true outcome class
        risk_score (array-like of shape (n_samples,) or (n_groups,)): overall or group level predicted risk scores
        keys: overall or subgroup level labels for labeling lines
        title_suffix: suffix for plot title
        pos_label: label identifying the positive class. Defaults to True.

    Returns:
        line plot of prediction bins X fraction of positive class
    """
    if not _check_array_of_arrays(y_true):
        y_true = [y_true]
        risk_score = [risk_score]
        keys = [keys]  # type: ignore

    fig, ax = plt.subplots()

    for j in range(len(y_true)):
        prob_true, prob_pred = calibration_curve(
            y_true[j],
            risk_score[j],
            n_bins=10,
            strategy="uniform",
            pos_label=pos_label,  # type: ignore
        )
        if lowess_frac:
            # When we create calibration curves with less data points (i.e. for subgroups),
            # it can look choppy and be difficult to interpret. We use locally weighted
            # scatterplot smoothing (LOWESS) to fit a smooth curve to the data points. LOWESS
            # is a non-parametric smoother that tries to find a curve of best fit without
            # assuming the data fits a particular shape.
            # Resource: https://www.statisticshowto.com/lowess-smoothing/
            prob_true = lowess(
                endog=prob_true,
                exog=prob_pred,
                frac=lowess_frac,
                it=0,
                is_sorted=False,
                return_sorted=False,
            )
        sns.lineplot(
            x=prob_pred, y=prob_true, color=PALETTE[j + 1], ax=ax, label=keys[j]
        )
    sns.lineplot(
        x=[0, 1],
        y=[0, 1],
        linestyle="dashed",
        color=PALETTE[0],
        ax=ax,
        label="Perfectly calibrated",
    )

    ax.set(
        xlabel="Mean predicted value",
        ylabel="Fraction of positives",
        title=f"Calibration Curve - {title_suffix}",
    )
    ax.set_xlim(left=-0.05, right=1.05)
    ax.set_ylim(bottom=-0.05, top=1.05)
    ax.legend(loc="lower right")
    return fig


def plot_support_score_histogram(
    support_scores: str | Sequence, title_suffix: str
) -> matplotlib.figure.Figure:
    """
    Plot histogram of support scores.

    Args:
        support_scores: support scores
        title_suffix: suffix for plot title
    """
    fig, ax = plt.subplots()
    sns.histplot(x=support_scores, ax=ax, color=PALETTE[1])
    ax.set(
        xlabel="Support Score", title=f"Distribution of support scores - {title_suffix}"
    )
    return fig


def plot_features_permutation_importance(
    result: sklearn.utils.Bunch, feature_cols: pd.Index
) -> matplotlib.figure.Figure:
    """
    Plot features' permutation importances (computed previously).

    Args:
        result
        feature_cols

    See Also:
        - :func:`compute_feature_permutation_importance()`
    """
    sorted_importances_idx = result.importances_mean.argsort()
    df_importances = pd.DataFrame(
        data=result.importances[sorted_importances_idx].T,
        columns=feature_cols[sorted_importances_idx],
    )

    fig, ax = plt.subplots(layout="constrained", figsize=(10, 10))
    df_importances.plot.box(vert=False, whis=10, ax=ax)
    ax.set_title("Permutation Feature Importances")
    ax.axvline(x=0, color="k", linestyle="--")
    ax.set_xlabel("decrease in model score")
    return fig


def plot_shap_beeswarm(shap_values):
    # Define the color palette
    start_color = (0.34, 0.79, 0.55)  # Green
    middle_color = (0.98, 0.82, 0.22)  # Yellow
    end_color = (0.95, 0.60, 0.19)  # Orange
    colors = [start_color, middle_color, end_color]
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors, 256)

    plt.figure(figsize=(5, 6))  # Adjust height as needed.

    # Plot the beeswarm
    ax = shap.plots.beeswarm(
        shap_values,
        show=False,
        max_display=25,
        color=cmap,
        color_bar=False,
        plot_size=(5, 25),
    )

    # Get the current y-axis tick labels (feature names)
    y_ticks_labels = [label.get_text() for label in ax.get_yticklabels()]

    # Modify the feature names
    modified_labels = [label.replace("_", " ").title() for label in y_ticks_labels]

    # Set the modified labels back to the y-axis
    ax.set_yticklabels(modified_labels)

    # Add horizontal lines
    # Get the y-axis tick positions
    y_ticks = ax.get_yticks()
    for y in y_ticks:
        # Add a dashed gray line
        ax.axhline(y=y, color="gray", linestyle="-", linewidth=0.5)

    # Add colorbar above chart
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(
            cmap=cmap,
        ),
        ax=ax,
        location="top",
        shrink=0.5,
        aspect=20,
        ticks=[],
        pad=0.02,
    )
    cbar.outline.set_visible(False)
    cbar.set_label("Feature Value", loc="center", labelpad=10)
    # Add labels to the left and right of the colorbar
    cbar.ax.text(
        -0.1,  # Adjust horizontal position for left label
        0.5,  # Vertical position (middle of colorbar)
        "Low",  # Left label text
        va="center",
        ha="right",
        transform=cbar.ax.transAxes,
    )
    cbar.ax.text(
        1.1,  # Adjust horizontal position for right label
        0.5,  # Vertical position (middle of colorbar)
        "High",  # Right label text
        va="center",
        ha="left",
        transform=cbar.ax.transAxes,
    )
    return plt.gcf()


def _check_array_of_arrays(input_array: pd.Series) -> bool:
    """
    Check if an input array contains sub-arrays, and return True if so; False otherwise.
    Used for plotting different groups of predictions.
    """
    try:
        assert isinstance(input_array, pd.Series)
        assert isinstance(input_array[0], list)
        return True
    except Exception:
        return False
