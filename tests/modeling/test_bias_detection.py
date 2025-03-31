import pytest
import pandas as pd
import numpy as np

from student_success_tool.modeling import bias_detection

np.random.seed(42)


@pytest.mark.parametrize(
    "targets, preds, expected_fnpr, expected_ci_lower, expected_ci_upper, valid_samples_flag",
    [
        (
            pd.Series(np.random.choice([False, True], size=500)),  # Use bool values
            pd.Series(np.random.choice([False, True], size=500)),
            0.515625,
            0.454406,
            0.57684,
            True
        ),
        (pd.Series([True] * 500, dtype=bool), pd.Series([True] * 500, dtype=bool), 0, 0, 0, False),
        (pd.Series([False] * 500, dtype=bool), pd.Series([False] * 500, dtype=bool), 0, 0, 0, False),
    ],
)
def test_calculate_fnpr_and_ci(
    targets, preds, expected_fnpr, expected_ci_lower, expected_ci_upper, valid_samples_flag,
):
    fnpr, ci_lower, ci_upper, valid_samples_flag = bias_detection.calculate_fnpr_and_ci(targets, preds)
    assert np.isclose(fnpr, expected_fnpr, equal_nan=True)
    assert np.isclose(ci_lower, expected_ci_lower, equal_nan=True)
    assert np.isclose(ci_upper, expected_ci_upper, equal_nan=True)


@pytest.mark.parametrize(
    "ci1, ci2, expected",
    [
        ((0.1, 0.3), (0.2, 0.4), True),
        ((0.1, 0.2), (0.3, 0.4), False),
        ((0.05, 0.15), (0.1, 0.2), True),
    ],
)
def test_check_ci_overlap(ci1, ci2, expected):
    assert bias_detection.check_ci_overlap(ci1, ci2) == expected


@pytest.mark.parametrize(
    "fnpr1, fnpr2, denom1, denom2, expected_p",
    [
        (0.2, 0.25, 100, 100, 0.3963327),
        (0.1, 0.15, 20, 35, np.nan),
        (0.3, 0.1, 200, 200, 2.4e-07),
    ],
)
def test_z_test_fnpr_difference(fnpr1, fnpr2, denom1, denom2, expected_p):
    p_value = bias_detection.z_test_fnpr_difference(fnpr1, fnpr2, denom1, denom2)
    assert np.isclose(p_value, expected_p, equal_nan=True)


@pytest.mark.parametrize(
    "group, sub1, sub2, percentage_difference, bias_type, split_name, flag, p, expected",
    [
        (
            "Gender",
            "Male",
            "Female",
            0.12,
            "Non-overlapping CIs",
            "train",
            "🔴 HIGH BIAS",
            0.005,
            {
                "group": "Gender",
                "subgroups": "Male vs Female",
                "percentage_difference": 12,
                "type": "Non-overlapping CIs, p-value: 0.005",
                "split_name": "train",
                "flag": "🔴 HIGH BIAS",
            },
        ),
    ],
)
def test_generate_bias_flag(group, sub1, sub2, percentage_difference, bias_type, split_name, flag, p, expected):
    assert (
        bias_detection.generate_bias_flag(
            group, sub1, sub2, percentage_difference, bias_type, split_name, flag, p
        )
        == expected
    )
