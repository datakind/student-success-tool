import os
import shutil
import typing as t
import uuid
from collections.abc import Sequence

import matplotlib.figure
import matplotlib.pyplot as plt
import mlflow
import mlflow.artifacts
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import recall_score
from statsmodels.nonparametric.smoothers_lowess import lowess

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


def create_risk_score_histogram(
    risk_score: str | Sequence, title_suffix: str
) -> matplotlib.figure.Figure:
    """
    Create histogram of risk scores

    Args:
        risk_score: risk scores
        title_suffix: suffix for plot title
    """
    fig, ax = plt.subplots()
    sns.histplot(x=risk_score, ax=ax, color=PALETTE[1])
    ax.set(xlabel="Risk Score", title=f"Distribution of risk scores - {title_suffix}")
    return fig


def check_array_of_arrays(input_array: pd.Series) -> bool:
    """
    Check if an input array contains sub-arrays. Used for plotting different
    groups of predictions

    Args:
        input_array

    Returns:
        True if the input_array contains sub-arrays
    """
    try:
        assert isinstance(input_array, pd.Series)
        assert isinstance(input_array[0], list)
        return True
    except Exception:
        return False


def create_calibration_curve(
    y_true: pd.Series,
    risk_score: pd.Series,
    keys: str | list[str],
    title_suffix: str,
    pos_label: PosLabelType,
    lowess_frac: t.Optional[float] = None,
) -> matplotlib.figure.Figure:
    """
    Create calibration plot

    Args:
        y_true (array-like of shape (n_samples,) or (n_groups,)): overall or group-level true outcome class
        risk_score (array-like of shape (n_samples,) or (n_groups,)): overall or group level predicted risk scores
        keys: overall or subgroup level labels for labeling lines
        title_suffix: suffix for plot title
        pos_label: label identifying the positive class. Defaults to True.

    Returns:
        line plot of prediction bins X fraction of positive class
    """
    if not check_array_of_arrays(y_true):
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

    result = recall_score(y_true, high_risk)
    assert isinstance(result, float)
    return result


def plot_sla_curve(
    y_true: pd.Series,
    risk_score: pd.Series,
    keys: str | list[str],
    title_suffix: str,
    pos_label: PosLabelType,
    alert_rates: list[float] = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
    label_alert_rate: float = 0.01,
) -> matplotlib.figure.Figure:
    """
    Create Sensitivity at Low Alert Rates plot

    Args:
        y_true (array-like of shape (n_samples,) or (n_groups,)): overall or group-level true outcome class
        risk_score (array-like of shape (n_samples,) or (n_groups,)): overall or group level predicted risk scores
        keys: overall or subgroup level labels for labeling lines
        title_suffix: suffix for plot title
        pos_label: label identifying the positive class in y_true
        alert_rates: alert rates to plot. Defaults to [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06].
        label_alert_rate: alert rate of interest to report sensitivity at. Defaults to 0.01.

    Returns:
        line plot of sensitivity at small alert rates
    """
    if not check_array_of_arrays(y_true):
        y_true = [y_true]
        risk_score = [risk_score]
        keys = [keys]  # type: ignore

    fig, ax = plt.subplots()

    for j in range(len(risk_score)):
        ss = []
        for i in alert_rates:  # calculate sensitivity at different alert rates
            s = get_sensitivity_of_top_q_pctl_thresh(
                y_true[j], risk_score[j], 1 - i, pos_label
            )
            ss.append(s)
        s_lab = round(
            get_sensitivity_of_top_q_pctl_thresh(
                y_true[j], risk_score[j], 1 - label_alert_rate, pos_label
            ),
            2,
        )
        ax.plot(
            alert_rates,
            ss,
            color=PALETTE[j + 1],
            label="{} (Sensitivity at {}% alert rate={})".format(
                keys[j], label_alert_rate * 100, s_lab
            ),
        )

    ax.set(
        xlabel="Alert rate",
        ylabel="sensitivity (true positive rate)",
        title=f"Sensitivity vs. Low Alert Rate - {title_suffix}",
    )
    ax.set_ylim(bottom=-0.02, top=1.02)
    ax.legend(loc="lower right")

    return fig


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


def compare_trained_models_plot(
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

    return fig


def create_evaluation_plots(
    data: pd.DataFrame,
    risk_score_col: str,
    y_true_col: str,
    pos_label: PosLabelType,
    split_type: str,
) -> tuple[matplotlib.figure.Figure, ...]:
    """
    Create plots to evaluate a model overall - risk score histogram,
    calibration curve, and sensitivity at low alert rates

    Args:
        data: containing predicted and actual outcome data
        risk_score_col: column name containing data of predicted risk scores
        y_true_col: column name containing data of actual outcome classes
        pos_label: label identifying the positive class in y_true
        split_type: type of data being plotted for labeling plots - train, test, or validation

    Returns:
        risk score histogram, calibration curve, and sensitivity at low alert rates figures
    """
    title_suffix = f"{split_type} data - Overall"
    hist_fig = create_risk_score_histogram(data[risk_score_col], title_suffix)
    cal_fig = create_calibration_curve(
        data[y_true_col], data[risk_score_col], "Overall", title_suffix, pos_label
    )
    sla_fig = plot_sla_curve(
        data[y_true_col], data[risk_score_col], "Overall", title_suffix, pos_label
    )
    return hist_fig, cal_fig, sla_fig


def create_evaluation_plots_by_subgroup(
    data: pd.DataFrame,
    risk_score_col: str,
    y_true_col: str,
    pos_label: PosLabelType,
    group_col: str,
    split_type: str,
) -> tuple[matplotlib.figure.Figure, ...]:
    """
    Create plots to evaluate a model by group - calibration curve
    and sensitivity at low alert rates

    Args:
        data: containing predicted and actual outcome data, as well as group label
        risk_score_col: column name containing data of predicted risk scores
        y_true_col: column name containing data of actual outcome classes
        pos_label: label identifying the positive class in y_true
        group_col: column name containing data of subgroup labels
        split_type: type of data being plotted for labeling plots - train, test, or validation

    Returns:
        calibration curve sensitivity at low alert rates figures by group
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

    sla_subgroup_plot = plot_sla_curve(ys, scores, names, title_suffix, pos_label)
    cal_subgroup_plot = create_calibration_curve(
        ys, scores, names, title_suffix, pos_label, lowess_frac=lowess_frac
    )

    return cal_subgroup_plot, sla_subgroup_plot
