import matplotlib.pyplot as plt
import mlflow.tracking
import numpy as np
import pandas as pd
import pytest

from student_success_tool.modeling import bias_detection

np.random.seed(42)

# HACK: log artifacts to tmp dir so they don't clutter repo
mlflow.set_tracking_uri("file:///tmp/mlflow")


@pytest.fixture
def patch_mlflow(monkeypatch):
    """Patch the mlflow.search_runs function."""

    def mock_search_runs():
        # This mock function does nothing and will be overridden by each test case
        pass

    monkeypatch.setattr(mlflow.tracking.MlflowClient, "search_runs", mock_search_runs)


@pytest.fixture
def mock_mlflow(monkeypatch):
    monkeypatch.setattr(mlflow, "log_figure", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        bias_detection, "log_group_metrics_to_mlflow", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        bias_detection, "log_subgroup_metrics_to_mlflow", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        bias_detection, "log_bias_flags_to_mlflow", lambda *args, **kwargs: None
    )


@pytest.fixture
def mock_helpers(monkeypatch):
    monkeypatch.setattr(bias_detection, "flag_bias", lambda fnpr_data: fnpr_data)
    monkeypatch.setattr(
        bias_detection, "plot_fnpr_group", lambda fnpr_data: plt.figure()
    )
    monkeypatch.setattr(
        bias_detection,
        "calculate_fnpr_and_ci",
        lambda y_true, y_pred: (0.5, 0.4, 0.6, sum(y_true)),
    )


def test_compute_group_bias_metrics(mock_helpers):
    df = pd.DataFrame(
        {
            "group_col": ["A", "A", "B", "B"],
            "target": [1, 0, 1, 0],
            "pred": [1, 0, 0, 1],
            "prob": [0.9, 0.2, 0.3, 0.8],
            "sample_weight_col": [1.2, 0.8, 1.2, 0.8],
        }
    )

    metrics, fnpr_data = bias_detection.compute_group_bias_metrics(
        split_data=df,
        split_name="test",
        group_col="group_col",
        target_col="target",
        pred_col="pred",
        pred_prob_col="prob",
        pos_label=1,
        sample_weight_col="sample_weight_col",
    )

    assert isinstance(metrics, list)
    assert isinstance(fnpr_data, list)
    assert all("FNPR" in m for m in metrics)
    assert all("subgroup" in f for f in fnpr_data)
    assert all(f["split_name"] == "test" for f in fnpr_data)
    assert all(f["fnpr"] > 0 and f["ci"][0] < f["fnpr"] < f["ci"][1] for f in fnpr_data)


@pytest.mark.parametrize(
    "targets, preds, expected_fnpr, expected_ci_lower, expected_ci_upper, valid_samples_flag",
    [
        (
            pd.Series(np.random.choice([False, True], size=500)),  # Use bool values
            pd.Series(np.random.choice([False, True], size=500)),
            0.504685,
            0.444092,
            0.565278,
            256,
        ),
        (
            pd.Series([True] * 500, dtype=bool),
            pd.Series([True] * 500, dtype=bool),
            0,
            0,
            0,
            500,
        ),
        (
            pd.Series([False] * 500, dtype=bool),
            pd.Series([False] * 500, dtype=bool),
            0,
            0,
            0,
            0,
        ),
    ],
)
def test_calculate_fnpr_and_ci(
    targets,
    preds,
    expected_fnpr,
    expected_ci_lower,
    expected_ci_upper,
    valid_samples_flag,
):
    fnpr, ci_lower, ci_upper, valid_samples_flag = bias_detection.calculate_fnpr_and_ci(
        targets, preds
    )
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
    "group, sub1, sub2, fnpr_percentage_difference, bias_type, split_name, flag, p, expected",
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
                "fnpr_percentage_difference": 0.12,
                "type": "Non-overlapping CIs, p-value: 0.005",
                "split_name": "train",
                "flag": "🔴 HIGH BIAS",
            },
        ),
    ],
)
def test_generate_bias_flag(
    group,
    sub1,
    sub2,
    fnpr_percentage_difference,
    bias_type,
    split_name,
    flag,
    p,
    expected,
):
    assert (
        bias_detection.generate_bias_flag(
            group,
            sub1,
            sub2,
            fnpr_percentage_difference,
            bias_type,
            split_name,
            flag,
            p,
        )
        == expected
    )
