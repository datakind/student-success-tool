import pandas as pd
import numpy as np
import scipy.stats as st
from sklearn.metrics import confusion_matrix

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


def calculate_fnpr_and_ci(
    targets: pd.Series,
    preds: pd.Series,
    apply_scaling: bool = True,
) -> tuple[float, float, float, bool]:
    """
    Calculates the False Negative Prediction Rate (FNPR) and its confidence interval, applying Laplace smoothing.

    Args:
        targets: Labels from model output
        preds: Predictions from model output
        min_fnpr_samples: Minimum number of true positives or false negatives for FNPR calculation.
        apply_scaling: Boolean of whether log scaling should be applied. We default to True since we want to
        sufficiently dampen FNPR variance in low sample sizes situations.

    Returns:
        fnpr: False Negative Parity Rate
        ci_min: Lower bound of the confidence interval
        ci_max: Upper bound of the confidence interval
        num_positives: We output the number of positives for reporting. Since when the number of positives are low (< MIN_FNPR_SAMPLES), then our FNPR computation may be not as reliable.
    """
    cm = confusion_matrix(targets, preds, labels=[False, True])
    tn, fp, fn, tp = cm.ravel()

    # Calculate FNPR & apply Log Scaling to smoothen FNPR at low sample sizes
    scaled_num_positives = fn + tp + np.log(fn + tp + 1) if apply_scaling else fn + tp
    fnpr = fn / scaled_num_positives if scaled_num_positives > 0 else 0

    # Confidence Interval Calculation
    margin = (
        Z * np.sqrt((fnpr * (1 - fnpr)) / scaled_num_positives)
        if scaled_num_positives > 0
        else 0
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
        "fnpr_percentage_difference": f"{fnpr_diff * 100:.2f}",
        "type": (
            bias_type
            if np.isnan(p_value)
            else f"{bias_type}, p-value: {'< 0.001' if p_value < 0.001 else f'{p_value:.3f}'}"
        ),
        "split_name": split_name,
        "flag": flag,
    }
    return flag_entry


def flag_bias(
    fnpr_data: list,
    split_name: str,
    high_bias_thresh: float = HIGH_FLAG_THRESHOLD,
    moderate_bias_thresh: float = MODERATE_FLAG_THRESHOLD,
    low_bias_thresh: float = LOW_FLAG_THRESHOLD,
    min_sample_ratio: float = 0.15,
) -> list[dict]:
    """
    Flags bias based on FNPR differences and confidence interval overlap.

    Args:
        fnpr_data: List of dictionaries containing FNPR and CI information for each subgroup.
        split_name: Name of the split (e.g. train/test/validate).
        high_bias_thresh: Threshold for flagging high bias.
        moderate_bias_thresh: Threshold for flagging moderate bias.
        low_bias_thresh: Threshold for flagging low bias.
        min_sample_ratio: Percentage of total positive samples required for valid FNPR comparison.
        We default to 15% since we want to ensure we are checking subgroups with sufficient data.

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
                            split_name,
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
                            split_name,
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
                                    split_name,
                                    flag,
                                    p_value,
                                )
                            )
                            break  # Exit after the first matched threshold

    return bias_flags
