import pandas as pd
import numpy as np
import matplotlib.figure
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import sklearn.metrics

# Z-score for 95% confidence interval
Z = st.norm.ppf(1 - (1 - 0.95) / 2)

# Flag FNPR difference thresholds
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

def evaluate_bias(
    run_id: str, 
    split_data: pd.DataFrame,
    split_name: str,
    student_group_cols: list,
    target_col: str, 
    pred_col: str, 
    pred_prob_col: str, 
    pos_label: str
) -> list:
    """
    Evaluates the bias in a model's predictions across different student groups for a split
    denoted by "split_name". For each student group, FNPR (False Negative Positive Rate) is
    computed and any detected biases are flagged. Then, the metrics & plots are logged to MLflow.

    Args:
        run_id (str): The ID of the MLflow run
        split_data (pd.DataFrame): Data for the current split to evaluate
        split_name (str): Name of the data split (e.g., "train", "test", or "val")
        student_group_cols (list): A list of columns representing student groups for bias analysis
        target_col (str): Column name for the target (actual) values
        pred_col (str): Column name for the model's predicted values
        pred_prob_col (str): Column name for the model's predicted probabilities
        pos_label (str or int): Label representing the positive class

    Returns:
        split_flags: List of flags for the data split
    """
    
    split_flags = []
    
    for group_col in student_group_cols:
        group_metrics, fnpr_data = compute_subgroup_bias_metrics(
            split_data,
            split_name,
            group_col,
            target_col, 
            pred_col, 
            pred_prob_col, 
            pos_label,
        )
        log_group_metrics_to_mlflow(group_metrics, split_name, group_col)
        
        # Detect bias flags
        bias_flags = flag_bias(fnpr_data)
        
        # Filter flags for groups where bias is detected
        group_flags = [flag for flag in bias_flags if flag["flag"] not in ["🟢 NO BIAS", "⚪ INSUFFICIENT DATA"]]
        
        if group_flags:
            fnpr_fig = fnpr_group_plot(fnpr_data)
            mlflow.log_figure(fnpr_fig, f"fnpr_plots/{split_name}_{group}_fnpr.png")
            
            for flag in group_flags:
                logging.info(
                    f"Run {run_id}: Bias detected in {flag['group']} - {flag['subgroups']} for split {split_name}. "
                    f"FNPR Difference: {flag['fnpr_percentage_difference']*100}% ({flag['type']}) [{flag['flag']}]"
                )
        
        split_flags.extend(group_flags)
    
    return split_flags


def compute_subgroup_bias_metrics(
    split_data: pd.DataFrame,
    split_name: str,
    group_col: str,
    target_col: str, 
    pred_col: str, 
    pred_prob_col: str, 
    pos_label: str, 
) -> tuple[dict, dict]:
    """
    Computes subgroup metrics (including FNPR) based on evaluation parameters and logs them to MLflow.

    Args:
        split_data (pd.DataFrame): Data for the current split to evaluate
        split_name (str): Name of the data split (e.g., "train", "test", or "val")
        target_col (str): Column name for the target variable.
        pred_col (str): Column name for the predictions.
        pred_prob_col (str): Column name for predicted probabilities.
        pos_label (str): Positive class label.
        student_group_cols (list): List of columns for subgroups.
    """
    group_metrics = []
    fnpr_data = []

    for subgroup, subgroup_data in split_data.groupby(group_col):
        labels = subgroup_data[target_col]
        preds = subgroup_data[pred_col]
        pred_probs = subgroup_data[pred_prob_col]
        
        fnpr, fnpr_lower, fnpr_upper, num_positives = (
            calculate_fnpr_and_ci(labels, preds)
        )

        fnpr_data.append(
            {
                "group": group,
                "subgroup": subgroup,
                "fnpr": fnpr,
                "split_name": split_name,
                "ci": (fnpr_lower, fnpr_upper),
                "size": len(subgroup_data),
                "number_of_positive_samples": num_positives,
            }
        )

        subgroup_metrics = {
            "Subgroup": subgroup,
            "Number of Samples": len(subgroup_data),
            "Number of Positive Samples": num_positives,
            "Actual Target Prevalence": round(labels.mean(), 2),
            "Predicted Target Prevalence": round(preds.mean(), 2),
            "FNPR": round(fnpr, 2),
            "FNPR CI Lower": round(fnpr_lower, 2),
            "FNPR CI Upper": round(fnpr_upper, 2),
            "Accuracy": round(sklearn.metrics.accuracy_score(labels, preds), 2),
            "Precision": round(
                sklearn.metrics.precision_score(
                    labels,
                    preds,
                    pos_label=pos_label,
                    zero_division=np.nan,
                ),
                2,
            ),
            "Recall": round(
                sklearn.metrics.recall_score(
                    labels,
                    preds,
                    pos_label=pos_label,
                    zero_division=np.nan,
                ),
                2,
            ),
            "Log Loss": round(
                sklearn.metrics.log_loss(
                    labels, pred_probs, labels=[False, True]
                ),
                2,
            ),
        }
        log_subgroup_metrics_to_mlflow(group_metrics, split_name, group_col)
        group_metrics.append(subgroup_metrics)

    return group_metrics, fnpr_data


def flag_bias(
    fnpr_data: list,
    high_bias_thresh: float = HIGH_FLAG_THRESHOLD,
    moderate_bias_thresh: float = MODERATE_FLAG_THRESHOLD,
    low_bias_thresh: float = LOW_FLAG_THRESHOLD,
    min_sample_ratio: float = 0.15,
) -> list[dict]:
    """
    Flags bias based on FNPR differences and confidence interval overlap.

    Args:
        fnpr_data: List of dictionaries containing FNPR and CI information for each subgroup.
        high_bias_thresh: Threshold for flagging high bias.
        moderate_bias_thresh: Threshold for flagging moderate bias.
        low_bias_thresh: Threshold for flagging low bias.
        min_sample_ratio: Percentage of total positive samples required for valid FNPR comparison.
        This gives us flexibility with smaller datasets. We default to 15% since we want to ensure
        we are checking subgroups with sufficient data. When calculating min_samples, we have an upper
        limit of 50 samples so that larger datasets aren't unnecessarily restricted.

    Returns:
        List of dictionaries with bias flag information.
    """
    total_group_positives = sum(
        subgroup["number_of_positive_samples"] for subgroup in fnpr_data
    )
    min_samples = min(50, int(min_sample_ratio * total_group_positives))

    bias_flags = []
    thresholds = [
        (high_bias_thresh, "🔴 HIGH BIAS", 0.01),
        (moderate_bias_thresh, "🟠 MODERATE BIAS", 0.01),
        (low_bias_thresh, "🟡 LOW BIAS", 0.1),
    ]

    for i, current in enumerate(fnpr_data):
        for other in fnpr_data[i + 1 :]:
            if current["fnpr"] > 0 and other["fnpr"] > 0:
                fnpr_diff = np.abs(current["fnpr"] - other["fnpr"])
                p_value = z_test_fnpr_difference(
                    current["fnpr"], other["fnpr"], current["size"], other["size"]
                )
                ci_overlap = check_ci_overlap(current["ci"], other["ci"])

                if np.isnan(p_value) or (
                    (current["number_of_positive_samples"] < min_samples)
                    or (other["number_of_positive_samples"] < min_samples)
                ):
                    bias_flags.append(
                        generate_bias_flag(
                            current["group"],
                            current["subgroup"],
                            other["subgroup"],
                            fnpr_diff,
                            "Insufficient samples for statistical test",
                            current["split_name"],
                            "⚪ INSUFFICIENT DATA",
                            p_value,
                        )
                    )
                elif fnpr_diff < low_bias_thresh or p_value > 0.1:
                    bias_flags.append(
                        generate_bias_flag(
                            current["group"],
                            current["subgroup"],
                            other["subgroup"],
                            fnpr_diff,
                            "No significant difference",
                            current["split_name"],
                            "🟢 NO BIAS",
                            p_value,
                        )
                    )
                else:
                    for threshold, flag, p_thresh in thresholds:
                        if fnpr_diff >= threshold and p_value <= p_thresh:
                            reason = (
                                "Overlapping CIs"
                                if ci_overlap
                                else "Non-overlapping CIs"
                            )
                            bias_flags.append(
                                generate_bias_flag(
                                    current["group"],
                                    current["subgroup"],
                                    other["subgroup"],
                                    fnpr_diff,
                                    reason,
                                    current["split_name"],
                                    flag,
                                    p_value,
                                )
                            )
                            break  # Exit after the first matched threshold

    return bias_flags


def calculate_fnpr_and_ci(
    targets: pd.Series,
    preds: pd.Series,
    apply_scaling: bool = True,
) -> tuple[float, float, float, bool]:
    """
    Calculates the False Negative Prediction Rate (FNPR) and its confidence interval, applying Log scaling.

    Args:
        targets: "Actual" labels from model output
        preds: Predictions from model output
        min_fnpr_samples: Minimum number of true positives or false negatives for FNPR calculation.
        apply_scaling: Boolean of whether log scaling should be applied. We default to True since we want to
        sufficiently dampen FNPR variance in low sample sizes situations.

    Returns:
        fnpr: False Negative Parity Rate
        ci_min: Lower bound of the confidence interval
        ci_max: Upper bound of the confidence interval
        num_positives: Number of positive samples, for reporting. When the number of positives is low, the FNPR computation may not be reliable.
    """
    cm = sklearn.metrics.confusion_matrix(targets, preds, labels=[False, True])
    tn, fp, fn, tp = cm.ravel()

    # Calculate FNPR & apply Log Scaling to smoothen FNPR at low sample sizes
    num_positives = fn + tp
    if apply_scaling:
        num_positives += np.log1p(num_positives)

    fnpr = fn / num_positives if num_positives > 0 else 0

    # Confidence Interval Calculation
    margin = (
        Z * np.sqrt((fnpr * (1 - fnpr)) / num_positives) if num_positives > 0 else 0
    )
    ci_min, ci_max = max(0, fnpr - margin), min(1, fnpr + margin)

    return fnpr, ci_min, ci_max, fn + tp


def check_ci_overlap(
    ci1: tuple[float, float],
    ci2: tuple[float, float],
) -> bool:
    """
    Checks whether confidence intervals (CIs) overlap. If they do, the FNPR differences
    are within the margin of error at the 95% confidence level. If the CIs do not
    overlap, this suggests strong statistical evidence that the FNPRs are different.

    Args:
        ci1: Confidence interval (min, max) for subgroup 1
        ci2: Confidence interval (min, max) for subgroup 2

    Returns:
        Boolean indicating whether the CIs overlap.
    """
    return not (ci1[1] < ci2[0] or ci2[1] < ci1[0])


def z_test_fnpr_difference(
    fnpr1: float,
    fnpr2: float,
    num_positives1: int,
    num_positives2: int,
) -> float:
    """
    Performs a z-test for the FNPR difference between two groups. If there are
    less than 30 samples of false negatives and true negatives, then we do not
    have enough data to perform a z-test. Thirty samples is the standard check
    for z-tests.

    Args:
        fnpr1: FNPR value for subgroup 1
        fnpr2: FNPR value for subgroup 2
        num_positives1: Number of false negatives + true negatives for subgroup 1
        num_positives2: Number of false negatives + true negatives for subgroup 2

    Returns:
        Two-tailed p-value for the z-test for the FNPR difference between the two subgroups.
    """
    if (
        num_positives1 <= 30 or num_positives2 <= 30
    ):  # Ensures valid sample sizes for z-test
        return np.nan
    std_error = np.sqrt(
        ((fnpr1 * (1 - fnpr1)) / num_positives1)
        + ((fnpr2 * (1 - fnpr2)) / num_positives2)
    )
    z_stat = (fnpr1 - fnpr2) / std_error
    return float(2 * (1 - st.norm.cdf(abs(z_stat))))  # Two-tailed p-value


def generate_bias_flag(
    group: str,
    subgroup1: str,
    subgroup2: str,
    fnpr_diff: float,
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
        fnpr_percentage_difference: Absolute value of percentage difference in FNPR
        bias_type: Type of bias (e.g. "Non-overlapping CIs", "Overlapping: : p-value: ...")
        split_name: Name of the split (e.g. train/test/validate)
        flag: Flag value (e.g. "🔴 HIGH BIAS", "🟠 MODERATE BIAS", "🟡 LOW BIAS", "🟢 NO BIAS")
        p_value: p-value for the z-test for the FNPR difference of the subgroup pair

    Returns:
        Dictionary containing bias flag information.
    """
    flag_entry = {
        "group": group,
        "subgroups": f"{subgroup1} vs {subgroup2}",
        "fnpr_percentage_difference": f"{round(fnpr, 2)}",
        "type": (
            bias_type
            if np.isnan(p_value)
            else f"{bias_type}, p-value: {'< 0.001' if p_value < 0.001 else f'{p_value:.3f}'}"
        ),
        "split_name": split_name,
        "flag": flag,
    }
    return flag_entry


def log_bias_flags_to_mlflow(all_model_flags: list):
    """
    Save and log bias flags to MLflow. If no flags exist for the model, then we do not log anything.
    
    Args:
        all_model_flags (list): Bias flags for across all splits
        (e.g. "train", "test", "val") of the model
    """
    if all_model_flags:
        df_model_flags = pd.DataFrame(all_model_flags)
        for flag in modeling.bias_detection.FLAG_NAMES.keys():
            flag_name = modeling.bias_detection.FLAG_NAMES[flag]
            df_flag = (
                df_all_flags[df_all_flags["flag"] == flag]
                .sort_values(by="fnpr_percentage_difference", key=lambda x: x.astype(float), ascending=False)
                if df_all_flags.shape[0] > 0
                else None
            )
            if df_flag is not None:
                bias_tmp_path = f"/tmp/{flag_name}_flags.csv"
                df_flag.to_csv(bias_tmp_path, index=False)
                mlflow.log_artifact(local_path=bias_tmp_path, artifact_path="bias_flags")


def log_group_metrics_to_mlflow(
    metrics: dict,
    split_name: str,
    group_col: str,
):
    """
    Saves and logs group-level bias metrics as a CSV artifact in MLflow.

    Args:
        metrics (dict): Dictionary containing computed group-level bias metrics.
        split_name (str): Name of the data split (e.g., "train", "test", "validation").
        group_col (str): Column name representing the group for bias evaluation.
    """
    df_group_metrics = pd.DataFrame(metrics)
    metrics_tmp_path = f"/tmp/{split_name}_{group_col}_metrics.csv"
    df_group_metrics.to_csv(metrics_tmp_path, index=False)
    mlflow.log_artifact(local_path=metrics_tmp_path, artifact_path="group_metrics")


def log_subgroup_metrics_to_mlflow(
    metrics: dict,
    split_name: str,
    group_col: str,
):
    """
    Logs individual subgroup-level metrics to MLflow.

    Args:
        metrics (dict): Dictionary of subgroup bias metrics.
        split_name (str): Name of the data split (e.g., "train", "test", "validation").
        group_col (str): Column name representing the group for bias evaluation.
    """
    for metric, value in metrics.items():
        if metric not in {"Subgroup", "Number of Samples"}:
            mlflow.log_metric(
                f"{split_name}_{group_col}_metrics/{metric}_subgroup", value
            )

def fnpr_group_plot(fnpr_data: list) -> matplotlib.figure.Figure:
    """
    Plots False Negative Prediction Rate (FNPR) for a group by subgroup on
    a split (train/test/val) of data with confidence intervals.

    Parameters:
    - fnpr_data: List with dictionaries for each subgroup. Each dictionary
    comprises of group, fnpr, ci (confidence interval), and split_name keys.
    """
    subgroups = [subgroup_data["subgroup"] for subgroup_data in fnpr_data]
    fnpr = [subgroup_data["fnpr"] for subgroup_data in fnpr_data]
    min_ci_errors = [
        subgroup_data["fnpr"] - subgroup_data["ci"][0] for subgroup_data in fnpr_data
    ]
    max_ci_errors = [
        subgroup_data["ci"][1] - subgroup_data["fnpr"] for subgroup_data in fnpr_data
    ]

    y_positions = list(range(len(subgroups)))
    fig, ax = plt.subplots()

    for i, (x, y, err_min, err_max) in enumerate(
        zip(fnpr, y_positions, min_ci_errors, max_ci_errors)
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
            label="FNPR" if i == 0 else "",
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(subgroups)
    ax.set_ylim(-1, len(subgroups))
    ax.set_ylabel("Subgroup")
    ax.set_xlabel("False Negative Parity Rate")
    ax.set_title(
        f"""FNPR @ 0.5 for {fnpr_data[0]["group"]} on {fnpr_data[0]["split_name"]}"""
    )
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    return fig
