import logging
import typing as t

import matplotlib.figure
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns
import sklearn.metrics

from . import evaluation

LOGGER = logging.getLogger(__name__)

# Z-score for 95% confidence interval
Z = st.norm.ppf(1 - (1 - 0.95) / 2)

# Flag FNR difference thresholds
HIGH_FLAG_THRESHOLD = 0.15
MODERATE_FLAG_THRESHOLD = 0.1
LOW_FLAG_THRESHOLD = 0.05

# Define flag types
FLAG_NAMES = {
    "⚪ INSUFFICIENT DATA": "insufficient_data",
    "🟢 NO BIAS": "no_bias",
    "🟡 LOW BIAS": "low_bias",
    "🟠 MODERATE BIAS": "moderate_bias",
    "🔴 HIGH BIAS": "high_bias",
}

# TODO: eventually we should use the custom_style.mplstyle colors, but currently
# the color palette does not have distinct enough colors for calibration by group
# where there are a lot of subgroups
PALETTE = sns.color_palette("Paired")

PosLabelType = t.Optional[int | float | bool | str]


def evaluate_bias(
    df_pred: pd.DataFrame,
    *,
    student_group_cols: list,
    pos_label: PosLabelType,
    split_col: str = "split",
    target_col: str = "target",
    pred_col: str = "pred",
    pred_prob_col: str = "pred_prob",
    sample_weight_col: str = "sample_weight_col",
) -> None:
    """
    Evaluates the bias in a model's predictions across different student groups for a split
    denoted by "split_name" across df_pred. For each student group, FNR (False Negative Rate)
    Parity is computed using absolute FNR percentage differences and any detected biases are
    flagged using hypothesis testing. Then, the metrics & plots are logged to MLflow.

    Args:
        run_id: The ID of the MLflow run
        df_pred: Pandas DataFrame with predictions from the model. The following columns
        need to be specified in df_pred: split_col, target_col, pred_col, and pred_probs_col.
        split_col: Column indicating split column ("train", "test", or "val")
        student_group_cols: A list of columns representing student groups for bias analysis
        target_col: Column name for the target (actual) values
        pred_col: Column name for the model's predicted values
        pred_prob_col: Column name for the model's predicted probabilities
        pos_label: Label representing the positive class
        sample_weight_col: Column name representing sample weights for the model.

    References:
        [1] https://fidelity.github.io/jurity/about_fairness.html
        [2] https://docs.oracle.com/en-us/iaas/tools/automlx/latest/legacy/v23.2.3/fairness.html
        [3] https://search.r-project.org/CRAN/refmans/fairness/html/fnr_parity.html
    """
    model_flags = []

    for split_name, split_data in df_pred.groupby(split_col):
        for group_col in student_group_cols:
            bias_metrics, perf_metrics, fnr_data = compute_group_bias_metrics(
                split_data,
                split_name,
                group_col,
                target_col,
                pred_col,
                pred_prob_col,
                pos_label,
                sample_weight_col,
            )

            log_group_metrics_to_mlflow(bias_metrics, f"bias_{split_name}", group_col)
            log_group_metrics_to_mlflow(perf_metrics, f"perf_{split_name}", group_col)

            # Detect bias flags
            all_flags = flag_bias(fnr_data)

            # Filter flags for groups where bias is detected
            group_flags = [
                flag
                for flag in all_flags
                if flag["flag"] not in ["🟢 NO BIAS", "⚪ INSUFFICIENT DATA"]
            ]

            if group_flags:
                fnr_fig = plot_fnr_group(fnr_data)
                mlflow.log_figure(
                    fnr_fig, f"fnr_plots/{split_name}_{group_col}_fnr.png"
                )
                plt.close()

                for flag in group_flags:
                    LOGGER.warning(
                        " Bias detected for %s on %s - %s, FNR Difference: %.2f%% (%s) [%s]",
                        flag["group"],
                        flag["split_name"],
                        flag["subgroups"],
                        flag["fnr_percentage_difference"] * 100,
                        flag["type"],
                        flag["flag"],
                    )

            model_flags.extend(all_flags)

    log_bias_flags_to_mlflow(model_flags)


def compute_group_bias_metrics(
    split_data: pd.DataFrame,
    split_name: str,
    group_col: str,
    target_col: str,
    pred_col: str,
    pred_prob_col: str,
    pos_label: PosLabelType,
    sample_weight_col: str,
) -> tuple[list, list, list]:
    """
    Computes group metrics (including FNR) based on evaluation parameters and logs them to MLflow.
    We split bias & performance metrics into separate dictionaries in our output to make the model
    card more readable with two separate tables.
    Args:
        split_data: Data for the current split to evaluate
        split_name: Name of the data split (e.g., "train", "test", or "val")
        target_col: Column name for the target variable.
        pred_col: Column name for the predictions.
        pred_prob_col: Column name for predicted probabilities.
        pos_label: Positive class label.
        student_group_cols: List of columns for subgroups.
    """
    bias_group_metrics = []
    perf_group_metrics = []
    fnr_data = []

    for subgroup_name, subgroup_data in split_data.groupby(group_col):
        labels = subgroup_data[target_col]
        preds = subgroup_data[pred_col]
        pred_probs = subgroup_data[pred_prob_col]

        fnr, fnr_lower, fnr_upper, num_positives = calculate_fnr_and_ci(labels, preds)

        fnr_subgroup_data = {
            "group": group_col,
            "subgroup": subgroup_name,
            "fnr": fnr,
            "split_name": split_name,
            "ci": (fnr_lower, fnr_upper),
            "size": len(subgroup_data),
            "number_of_positive_samples": num_positives,
        }

        eval_metrics = evaluation.compute_classification_perf_metrics(
            labels,
            preds,
            pred_probs,
            pos_label=pos_label,
            sample_weights=(
                subgroup_data[sample_weight_col]
                if sample_weight_col in subgroup_data.columns
                else None
            ),
        )
        # HACK: avoid duplicative metrics
        eval_metrics.pop("num_positives", None)
        bias_subgroup_metrics, perf_subgroup_metrics = format_subgroup_metrics(
            eval_metrics, fnr_subgroup_data
        )

        log_subgroup_metrics_to_mlflow(
            bias_subgroup_metrics, f"{split_name}_bias", group_col
        )
        log_subgroup_metrics_to_mlflow(
            perf_subgroup_metrics, f"{split_name}_performance", group_col
        )

        bias_group_metrics.append(bias_subgroup_metrics)
        perf_group_metrics.append(perf_subgroup_metrics)
        fnr_data.append(fnr_subgroup_data)

    return bias_group_metrics, perf_group_metrics, fnr_data


def format_subgroup_metrics(
    eval_metrics: dict, fnr_subgroup_data: dict
) -> tuple[dict, dict]:
    """
    Formats the evaluation metrics and bias metrics together for logging into MLflow.

    Args:
        eval_metrics: Dictionary performance metrics for subgroup
        fnr_subgroup_data: List of dictionaries containing FNR and CI information for each subgroup.
    Returns:
        Tuple of dictionaries summarizing both performance & bias metrics
    """
    bias_metrics = {
        "Subgroup": fnr_subgroup_data["subgroup"],
        "Number of Samples": eval_metrics["num_samples"],
        "Number of Positive Samples": fnr_subgroup_data["number_of_positive_samples"],
        "Actual Target Prevalence": round(eval_metrics["true_positive_prevalence"], 2),
        "Predicted Target Prevalence": round(
            eval_metrics["pred_positive_prevalence"], 2
        ),
        # Bias Metrics
        "FNR": round(fnr_subgroup_data["fnr"], 2),
        "FNR CI Lower": round(fnr_subgroup_data["ci"][0], 2),
        "FNR CI Upper": round(fnr_subgroup_data["ci"][1], 2),
    }
    performance_metrics = {
        "Subgroup": fnr_subgroup_data["subgroup"],
        # Performance Metrics
        "Accuracy": round(eval_metrics["accuracy"], 2),
        "Precision": round(eval_metrics["precision"], 2),
        "Recall": round(eval_metrics["recall"], 2),
        "F1 Score": round(eval_metrics["f1_score"], 2),
        "Log Loss": round(eval_metrics["log_loss"], 2),
    }
    return bias_metrics, performance_metrics


def flag_bias(
    fnr_data: list,
    high_bias_thresh: float = HIGH_FLAG_THRESHOLD,
    moderate_bias_thresh: float = MODERATE_FLAG_THRESHOLD,
    low_bias_thresh: float = LOW_FLAG_THRESHOLD,
    min_sample_ratio: float = 0.15,
) -> list[dict]:
    """
    Flags bias based on FNR Parity (via absolute FNR percentage differences) and confidence interval overlap.

    Args:
        fnr_data: List of dictionaries containing FNR and CI information for each subgroup.
        high_bias_thresh: Threshold for flagging high bias.
        moderate_bias_thresh: Threshold for flagging moderate bias.
        low_bias_thresh: Threshold for flagging low bias.
        min_sample_ratio: Percentage of total positive samples required for valid FNR comparison.
        This gives us flexibility with smaller datasets. We default to 15% since we want to ensure
        we are checking subgroups with sufficient data. When calculating min_samples, we have an upper
        limit of 50 samples so that larger datasets aren't unnecessarily restricted.

    Returns:
        List of dictionaries with bias flag information.
    """
    total_group_positives = sum(
        subgroup["number_of_positive_samples"] for subgroup in fnr_data
    )
    min_samples = min(50, int(min_sample_ratio * total_group_positives))

    bias_flags = []
    thresholds = [
        (high_bias_thresh, "🔴 HIGH BIAS", 0.01),
        (moderate_bias_thresh, "🟠 MODERATE BIAS", 0.01),
        (low_bias_thresh, "🟡 LOW BIAS", 0.1),
    ]

    for i, current in enumerate(fnr_data):
        for other in fnr_data[i + 1 :]:
            if current["fnr"] > 0 and other["fnr"] > 0:
                # Determine ordering based on FNR values
                sg1, sg2 = (
                    (current, other)
                    if current["fnr"] >= other["fnr"]
                    else (other, current)
                )
                # Guaranteed to be greater than zero
                fnr_diff = sg1["fnr"] - sg2["fnr"]
                p_value = z_test_fnr_difference(
                    sg1["fnr"], sg2["fnr"], sg1["size"], sg2["size"]
                )
                ci_overlap = check_ci_overlap(sg1["ci"], sg2["ci"])

                if np.isnan(p_value) or (
                    (sg1["number_of_positive_samples"] < min_samples)
                    or (sg2["number_of_positive_samples"] < min_samples)
                ):
                    bias_flags.append(
                        generate_bias_flag(
                            sg1["group"],
                            sg1["subgroup"],
                            sg2["subgroup"],
                            fnr_diff,
                            "insufficient samples for statistical test",
                            sg1["split_name"],
                            "⚪ INSUFFICIENT DATA",
                            p_value,
                        )
                    )
                elif fnr_diff < low_bias_thresh or p_value > 0.1:
                    bias_flags.append(
                        generate_bias_flag(
                            sg1["group"],
                            sg1["subgroup"],
                            sg2["subgroup"],
                            fnr_diff,
                            "no significant difference",
                            sg1["split_name"],
                            "🟢 NO BIAS",
                            p_value,
                        )
                    )
                else:
                    for threshold, flag, p_thresh in thresholds:
                        if fnr_diff >= threshold and p_value <= p_thresh:
                            reason = (
                                "overlapping confidence intervals"
                                if ci_overlap
                                else "non-overlapping confidence intervals"
                            )
                            bias_flags.append(
                                generate_bias_flag(
                                    sg1["group"],
                                    sg1["subgroup"],
                                    sg2["subgroup"],
                                    fnr_diff,
                                    reason,
                                    sg1["split_name"],
                                    flag,
                                    p_value,
                                )
                            )
                            break  # Exit after the first matched threshold

    return bias_flags


def calculate_fnr_and_ci(
    targets: pd.Series,
    preds: pd.Series,
    apply_scaling: bool = True,
) -> tuple[float, float, float, bool]:
    """
    Calculates the False Negative Rate (FNR) and its confidence interval, applying Log scaling.

    Args:
        targets: "Actual" labels from model output
        preds: Predictions from model output
        min_fnr_samples: Minimum number of true positives or false negatives for FNR calculation.
        apply_scaling: Boolean of whether log scaling should be applied. We default to True since we want to
        sufficiently dampen FNR variance in low sample sizes situations.

    Returns:
        fnr: False Negative Rate
        ci_min: Lower bound of the confidence interval
        ci_max: Upper bound of the confidence interval
        num_positives: Number of positive samples, for reporting. When the number of positives is low, the FNR computation may not be reliable.
    """
    cm = sklearn.metrics.confusion_matrix(targets, preds, labels=[False, True])
    tn, fp, fn, tp = cm.ravel()

    # Calculate FNR & apply Log Scaling to smoothen FNR at low sample sizes
    num_positives = fn + tp
    if apply_scaling:
        num_positives += np.log1p(num_positives)

    fnr = fn / num_positives if num_positives > 0 else 0

    # Confidence Interval Calculation
    margin = Z * np.sqrt((fnr * (1 - fnr)) / num_positives) if num_positives > 0 else 0
    ci_min, ci_max = max(0, fnr - margin), min(1, fnr + margin)

    return fnr, ci_min, ci_max, fn + tp


def check_ci_overlap(
    ci1: tuple[float, float],
    ci2: tuple[float, float],
) -> bool:
    """
    Checks whether confidence intervals (CIs) overlap. If they do, the FNR differences
    are within the margin of error at the 95% confidence level. If the CIs do not
    overlap, this suggests strong statistical evidence that the FNRs are different.

    Args:
        ci1: Confidence interval (min, max) for subgroup 1
        ci2: Confidence interval (min, max) for subgroup 2

    Returns:
        Boolean indicating whether the CIs overlap.
    """
    return not (ci1[1] < ci2[0] or ci2[1] < ci1[0])


def z_test_fnr_difference(
    fnr1: float,
    fnr2: float,
    num_positives1: int,
    num_positives2: int,
) -> float:
    """
    Performs a z-test for the FNR difference between two groups. If there are
    less than 30 samples of false negatives and true negatives, then we do not
    have enough data to perform a z-test. Thirty samples is the standard check
    for z-tests.

    Args:
        fnr1: FNR value for subgroup 1
        fnr2: FNR value for subgroup 2
        num_positives1: Number of false negatives + true negatives for subgroup 1
        num_positives2: Number of false negatives + true negatives for subgroup 2

    Returns:
        Two-tailed p-value for the z-test for the raw FNR difference between the two subgroups.
    """
    if (
        num_positives1 <= 30 or num_positives2 <= 30
    ):  # Ensures valid sample sizes for z-test
        return np.nan
    std_error = np.sqrt(
        ((fnr1 * (1 - fnr1)) / num_positives1) + ((fnr2 * (1 - fnr2)) / num_positives2)
    )
    z_stat = (fnr1 - fnr2) / std_error
    return float(2 * (1 - st.norm.cdf(abs(z_stat))))  # Two-tailed p-value


def generate_bias_flag(
    group: str,
    subgroup1: str,
    subgroup2: str,
    fnr_percentage_difference: float,
    bias_type: str,
    split_name: str,
    flag: str,
    p_value: float = np.nan,
) -> dict:
    """
    Aggregate bias flag information for a given subgroup pair into a dict.

    Args:
        group: Name of the group (e.g. "Gender", "Race", "Age")
        subgroup1: Name of the subgroup 1 (e.g. "Female", "Male", "Asian", "African American", "Caucasian")
        subgroup2: Name of the subgroup 2 (e.g. "Female", "Male", "Asian", "African American", "Caucasian")
        fnr_percentage_difference: Absolute value of percentage difference in FNR
        bias_type: Type of bias (e.g. "Non-overlapping CIs", "Overlapping: : p-value: ...")
        split_name: Name of the split (e.g. train/test/validate)
        flag: Flag value (e.g. "🔴 HIGH BIAS", "🟠 MODERATE BIAS", "🟡 LOW BIAS", "🟢 NO BIAS")
        p_value: p-value for the z-test for the FNR difference of the subgroup pair

    Returns:
        Dictionary containing bias flag information.
    """
    flag_entry = {
        "group": group,
        "subgroups": f"{subgroup1} vs {subgroup2}",
        "fnr_percentage_difference": round(fnr_percentage_difference, 4),
        "type": (
            bias_type
            if np.isnan(p_value)
            else f"{bias_type} with a p-value {'less than 0.001' if p_value < 0.001 else f'of {p_value:.3f}'}"
        ),
        "split_name": split_name,
        "flag": flag,
    }
    return flag_entry


def log_bias_flags_to_mlflow(all_model_flags: list) -> None:
    """
    Save and log bias flags to MLflow. If no flags exist for the model, then we do not log anything.

    Args:
        all_model_flags: Bias flags for across all splits
        (e.g. "train", "test", "val") of the model
    """
    if all_model_flags:
        df_model_flags = pd.DataFrame(all_model_flags)
        for flag in FLAG_NAMES.keys():
            flag_name = FLAG_NAMES[flag]
            df_flag = (
                df_model_flags[df_model_flags["flag"] == flag].sort_values(
                    by="fnr_percentage_difference", ascending=False
                )
                if df_model_flags.shape[0] > 0
                else None
            )
            if df_flag is not None:
                bias_tmp_path = f"/tmp/{flag_name}_flags.csv"
                df_flag.to_csv(bias_tmp_path, index=False)
                mlflow.log_artifact(
                    local_path=bias_tmp_path, artifact_path="bias_flags"
                )


def log_group_metrics_to_mlflow(
    group_metrics: list,
    split_name: str,
    group_col: str,
) -> None:
    """
    Saves and logs group-level bias metrics as a CSV artifact in MLflow.

    Args:
        group_metrics: List of dictionaries containing computed group-level bias
        or performance metrics.
        split_name: Name of the data split (e.g., "train", "test", "validation").
        group_col: Column name representing the group for bias evaluation.
    """
    df_group_metrics = pd.DataFrame(group_metrics)
    metrics_tmp_path = f"/tmp/{split_name}_{group_col}_metrics.csv"
    df_group_metrics.to_csv(metrics_tmp_path, index=False)
    mlflow.log_artifact(local_path=metrics_tmp_path, artifact_path="group_metrics")


def log_subgroup_metrics_to_mlflow(
    subgroup_metrics: dict,
    split_name: str,
    group_col: str,
) -> None:
    """
    Logs individual subgroup-level metrics to MLflow.

    Args:
        subgroup_metrics: Dictionary of subgroup bias metrics.
        split_name: Name of the data split (e.g., "train", "test", "validation").
        group_col: Column name representing the group for bias evaluation.
    """
    for metric, value in subgroup_metrics.items():
        if metric not in {"Subgroup", "Number of Samples"}:
            mlflow.log_metric(
                f"{split_name}_{group_col}_metrics/{metric}_subgroup", value
            )


def plot_fnr_group(fnr_data: list) -> matplotlib.figure.Figure:
    """
    Plots False Negative Rate (FNR) for a group by subgroup on
    a split (train/test/val) of data with confidence intervals.

    Args:
        fnr_data: List with dictionaries for each subgroup. Each dictionary
        comprises of group, fnr, ci (confidence interval), and split_name keys.
    """
    subgroups = [subgroup_data["subgroup"] for subgroup_data in fnr_data]
    fnr = [subgroup_data["fnr"] for subgroup_data in fnr_data]
    min_ci_errors = [
        subgroup_data["fnr"] - subgroup_data["ci"][0] for subgroup_data in fnr_data
    ]
    max_ci_errors = [
        subgroup_data["ci"][1] - subgroup_data["fnr"] for subgroup_data in fnr_data
    ]

    y_positions = list(range(len(subgroups)))
    fig, ax = plt.subplots()

    for i, (x, y, err_min, err_max) in enumerate(
        zip(fnr, y_positions, min_ci_errors, max_ci_errors)
    ):
        color = PALETTE[i % len(PALETTE)]
        ax.errorbar(
            x=x,
            y=y,
            xerr=[[err_min], [err_max]],
            fmt="o",
            capsize=5,
            capthick=2,
            markersize=8,
            linewidth=2,
            color=color,
            label="FNR" if i == 0 else "",
        )

    ax.set(
        yticks=y_positions,
        yticklabels=subgroups,
        ylim=(-1, len(subgroups)),
        ylabel="Subgroup",
        xlabel="False Negative Rate",
        title=f"""FNR @ 0.5 for {fnr_data[0]["group"]} on {fnr_data[0]["split_name"]}""",
    )
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    return fig
