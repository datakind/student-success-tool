import pandas as pd
import numpy as np
import scipy.stats as st
from sklearn.metrics import confusion_matrix

# FNPR sample threshold
MIN_FNPR_SAMPLES = 50
Z = st.norm.ppf(1 - (1 - 0.95) / 2)

# Flag FNPR difference thresholds
HIGH_FLAG_THRESHOLD = 0.15
MODERATE_FLAG_THRESHOLD = 0.1
LOW_FLAG_THRESHOLD = 0.05

# Define flag types
FLAG_NAMES = {
    "🟢 NO BIAS": "no_bias",
    "🟡 LOW BIAS": "low_bias",
    "🟠 MODERATE BIAS": "moderate_bias",
    "🔴 HIGH BIAS": "high_bias",
}


def calculate_fnpr_and_ci(
    targets: pd.Series,
    preds: pd.Series,
    min_fnpr_samples: int = MIN_FNPR_SAMPLES,
) -> tuple[float, float, float]:
    """
    Calculates the False Negative Prediction Rate (FNPR) and its confidence interval.

    Args:
        targets: Labels from model output
        preds: Predictions from model output
        min_fnpr_samples: Minimum number of true positives or false negatives for FNPR calculation. When TP or FN are low, FNPR can be very unstable and flag subgroups based on small differences in TP or FN.
    """
    cm = confusion_matrix(targets, preds, labels=[False, True])
    tn, fp, fn, tp = (
        cm.ravel()
        if cm.shape == (2, 2)
        else np.bincount(np.array(targets) * 2 + np.array(preds), minlength=4)
    )

    denominator = fn + tp

    if (tp < min_fnpr_samples) or (fn < min_fnpr_samples):
        return np.nan, np.nan, np.nan

    fnpr = fn / denominator
    margin = Z * np.sqrt((fnpr * (1 - fnpr)) / denominator)
    return fnpr, max(0, fnpr - margin), min(1, fnpr + margin)


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
    """
    return not (ci1[1] < ci2[0] or ci2[1] < ci1[0])


def z_test_fnpr_difference(
    fnpr1: float,
    fnpr2: float,
    denominator1: int,
    denominator2: int,
) -> float:
    """
    Performs a z-test for the FNPR difference between two groups. If there are
    less than 30 samples of false negatives and true negatives, then we do not
    have enough data to perform a z-test. Thirty samples is the standard check
    for z-tests.

    Args:
        fnpr1: FNPR value for subgroup 1
        fnpr2: FNPR value for subgroup 2
        denominator1: Number of false negatives + true negatives for subgroup 1
        denominator2: Number of false negatives + true negatives for subgroup 2
    """
    if (
        denominator1 <= 30 or denominator2 <= 30
    ):  # Ensures valid sample sizes for z-test
        return np.nan
    std_error = np.sqrt(
        (fnpr1 * (1 - fnpr1)) / denominator1 + (fnpr2 * (1 - fnpr2)) / denominator2
    )
    z_stat = (fnpr1 - fnpr2) / std_error
    return float(2 * (1 - st.norm.cdf(abs(z_stat))))  # Two-tailed p-value


def log_bias_flag(
    group: str,
    subgroup1: str,
    subgroup2: str,
    fnpr_diff: float,
    bias_type: str,
    dataset: str,
    flag: str,
    p_value: float = np.nan,
) -> dict:
    """
    Aggregate bias flag information for a given subgroup pair into a dict.

    Args:
        group: Name of the group (e.g. "Gender", "Race", "Age")
        subgroup1: Name of the subgroup 1 (e.g. "Female", "Male", "Asian", "African American", "Caucasian")
        subgroup2: Name of the subgroup 2 (e.g. "Female", "Male", "Asian", "African American", "Caucasian")
        fnpr_diff: Absolute value of difference in FNPR
        bias_type: Type of bias (e.g. "Non-overlapping CIs", "Overlapping: : p-value: ...")
        dataset: Name of the dataset (e.g. train/test/validate)
        flag: Flag value (e.g. "🔴 HIGH BIAS", "🟠 MODERATE BIAS", "🟡 LOW BIAS", "🟢 NO BIAS")
        p_value: p-value for the z-test for the FNPR difference of the subgroup pair
    """
    flag_entry = {
        "group": group,
        "subgroups": f"{subgroup1} vs {subgroup2}",
        "difference": fnpr_diff * 100,
        "type": bias_type
        if np.isnan(p_value)
        else f"{bias_type}, p-value: {'< 0.001' if p_value < 0.001 else f'{p_value:.3f}'}",
        "dataset": dataset,
        "flag": flag,
    }
    return flag_entry


def flag_bias(
    fnpr_data: list,
    split_name: str,
    high_bias_thresh: float = HIGH_FLAG_THRESHOLD,
    moderate_bias_thresh: float = MODERATE_FLAG_THRESHOLD,
    low_bias_thresh: float = LOW_FLAG_THRESHOLD,
) -> list[dict]:
    """
    Flags bias based on FNPR differences and confidence interval overlap.

    Args:
        fnpr_data: List of dictionaries containing FNPR and CI information for each subgroup.
        split_name: Name of the dataset (e.g. train/test/validate).
        high_bias_thresh: Threshold for flagging high bias.
        moderate_bias_thresh: Threshold for flagging moderate bias.
        low_bias_thresh: Threshold for flagging low bias.
    Returns:
        List of dictionaries with bias flag information.
    """
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

                if (fnpr_diff < low_bias_thresh) or (p_value > 0.1):
                    bias_flags.append(
                        log_bias_flag(
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
                        if fnpr_diff >= threshold:
                            reason = (
                                "Overlapping CIs"
                                if ci_overlap and p_value and p_value < p_thresh
                                else "Non-overlapping CIs"
                            )
                            bias_flags.append(
                                log_bias_flag(
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
