import pytest
import pandas as pd
import numpy as np

from student_success_tool.modeling import bias_detection

np.random.seed(42)

@pytest.mark.parametrize(
    "targets, preds, expected_fnpr, expected_ci_lower, expected_ci_upper",
    [
        (pd.Series(np.random.choice([0, 1], size=500)),
         pd.Series(np.random.choice([0, 1], size=500)),
         0.515625, 0.454406, 0.57684),
        (pd.Series(np.ones(500)), pd.Series(np.ones(500)), np.nan, np.nan, np.nan),
        (pd.Series(np.zeros(500)), pd.Series(np.zeros(500)),, np.nan, np.nan, np.nan),
    ]
)
def test_calculate_fnpr_and_ci(targets, preds, expected_fnpr, expected_ci_lower, expected_ci_upper):
    fnpr, ci_lower, ci_upper = bias_detection.calculate_fnpr_and_ci(targets, preds)
    assert np.isclose(fnpr, expected_fnpr, equal_nan=True)
    assert np.isclose(ci_lower, expected_ci_lower, equal_nan=True)
    assert np.isclose(ci_upper, expected_ci_upper, equal_nan=True)

@pytest.mark.parametrize(
    "ci1, ci2, expected",
    [
        ((0.1, 0.3), (0.2, 0.4), True),
        ((0.1, 0.2), (0.3, 0.4), False),
        ((0.05, 0.15), (0.1, 0.2), True),
    ]
)
def test_check_ci_overlap(ci1, ci2, expected):
    assert bias_detection.check_ci_overlap(ci1, ci2) == expected

@pytest.mark.parametrize(
    "fnpr1, fnpr2, denom1, denom2, expected_p",
    [
        (0.2, 0.25, 100, 100, 0.3),
        (0.1, 0.15, 50, 50, None),  # Sample size too small
        (0.3, 0.1, 200, 200, 0.001),
    ]
)
def test_z_test_fnpr_difference(fnpr1, fnpr2, denom1, denom2, expected_p):
    p_value = bias_detection.z_test_fnpr_difference(fnpr1, fnpr2, denom1, denom2)
    assert (p_value is None and expected_p is None) or np.isclose(p_value, expected_p, atol=0.05)

@pytest.mark.parametrize(
    "group, sub1, sub2, diff, bias_type, dataset, flag, p, expected",
    [
        ("Gender", "Male", "Female", 0.12, "Non-overlapping CIs", "train", "🔴 HIGH BIAS", 0.005,
         {"group": "Gender", "subgroups": "Male vs Female", "difference": 12, "type": "Non-overlapping CIs, p-value: < 0.001", "dataset": "train", "flag": "🔴 HIGH BIAS"}),
    ]
)
def test_log_bias_flag(group, sub1, sub2, diff, bias_type, dataset, flag, p, expected):
    assert bias_detection.log_bias_flag(group, sub1, sub2, diff, bias_type, dataset, flag, p) == expected

@pytest.mark.parametrize(
    "fnpr_data, split_name, expected_length",
    [
        ([{"group": "Gender", "subgroup": "Male", "fnpr": 0.15, "ci": (0.1, 0.2), "size": 100},
          {"group": "Gender", "subgroup": "Female", "fnpr": 0.1, "ci": (0.05, 0.15), "size": 100}], "test", 1),
    ]
)
def test_flag_bias(fnpr_data, split_name, expected_length):
    bias_flags = bias_detection.flag_bias(fnpr_data, split_name)
    assert len(bias_flags) == expected_length
