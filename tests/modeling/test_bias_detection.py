import matplotlib.pyplot as plt
import mlflow.tracking
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

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
    monkeypatch.setattr(bias_detection, "flag_bias", lambda fnr_data: fnr_data)
    monkeypatch.setattr(bias_detection, "plot_fnr_group", lambda fnr_data: plt.figure())
    monkeypatch.setattr(
        bias_detection,
        "calculate_fnr_and_ci",
        lambda y_true, y_pred: (0.5, 0.4, 0.6, sum(y_true)),
    )


@pytest.fixture
def mock_df_pred():
    return pd.DataFrame(
        {
            "split": ["test"] * 6,
            "Gender": ["Male", "Female", "Female", "Male", "Male", "Female"],
            "target": [1, 1, 0, 0, 1, 0],
            "pred": [1, 0, 0, 0, 1, 1],
            "pred_prob": [0.9, 0.3, 0.2, 0.4, 0.8, 0.7],
        }
    )


def test_evaluate_bias_basic(mock_df_pred):
    with (
        patch("student_success_tool.modeling.bias_detection.mlflow"),
        patch(
            "student_success_tool.modeling.bias_detection.plot_fnr_group"
        ) as mock_plot_fnr,
        patch(
            "student_success_tool.modeling.bias_detection.flag_bias"
        ) as mock_flag_bias,
        patch(
            "student_success_tool.modeling.bias_detection.log_bias_scores_to_mlflow"
        ) as mock_log_scores,
        patch(
            "student_success_tool.modeling.bias_detection.log_group_metrics_to_mlflow"
        ) as mock_log_group,
        patch(
            "student_success_tool.modeling.bias_detection.log_subgroup_metrics_to_mlflow"
        ) as mock_log_subgroup,
        patch(
            "student_success_tool.modeling.bias_detection.log_bias_flags_to_mlflow"
        ) as mock_log_flags,
    ):
        mock_flag_bias.return_value = [
            {
                "group": "Gender",
                "subgroups": "Female vs Male",
                "flag": "🟠 MODERATE BIAS",
                "fnr_percentage_difference": 0.12,
                "type": "non-overlapping confidence intervals",
                "split_name": "test",
                "p_value": 0.005,
            },
            {
                "group": "Gender",
                "subgroups": "Male vs Female",
                "flag": "🟢 NO BIAS",
                "fnr_percentage_difference": 0.02,
                "type": "no significant difference",
                "split_name": "test",
                "p_value": 0.6,
            },
        ]

        # Run
        bias_detection.evaluate_bias(
            df_pred=mock_df_pred,
            student_group_cols=["Gender"],
            pos_label=1,
            target_col="target",
            pred_col="pred",
            pred_prob_col="pred_prob",
            sample_weight_col="",
        )

        # -- Validate --
        assert mock_flag_bias.called
        assert mock_log_group.called
        assert mock_log_subgroup.called
        assert mock_log_scores.called
        assert mock_log_flags.called

        # Validate plotting
        mock_plot_fnr.assert_called_once()

        # -- Bias score checks --
        args, kwargs = mock_log_scores.call_args
        bias_score_summary = args[0]  # the first positional argument is the scores dict

        assert "bias_score_sum" in bias_score_summary
        assert "bias_score_mean" in bias_score_summary
        assert bias_score_summary["bias_score_mean"] > 0
        assert (
            bias_score_summary["num_valid_comparisons"] == 2
        )  # 1 moderate + 1 no bias
        assert bias_score_summary["num_bias_flags"] == 1  # only 1 moderate flag


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

    bias_metrics, perf_metrics, fnr_data = bias_detection.compute_group_bias_metrics(
        split_data=df,
        split_name="test",
        group_col="group_col",
        target_col="target",
        pred_col="pred",
        pred_prob_col="prob",
        pos_label=1,
        sample_weight_col="sample_weight_col",
    )

    assert isinstance(bias_metrics, list)
    assert isinstance(perf_metrics, list)
    assert isinstance(fnr_data, list)
    assert all("FNR" in m for m in bias_metrics)
    assert all("subgroup" in f for f in fnr_data)
    assert all(f["split_name"] == "test" for f in fnr_data)
    assert all(f["fnr"] > 0 and f["ci"][0] < f["fnr"] < f["ci"][1] for f in fnr_data)


@pytest.mark.parametrize(
    "targets, preds, expected_fnr, expected_ci_lower, expected_ci_upper, valid_samples_flag",
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
def test_calculate_fnr_and_ci(
    targets,
    preds,
    expected_fnr,
    expected_ci_lower,
    expected_ci_upper,
    valid_samples_flag,
):
    fnr, ci_lower, ci_upper, valid_samples_flag = bias_detection.calculate_fnr_and_ci(
        targets, preds
    )
    assert np.isclose(fnr, expected_fnr, equal_nan=True)
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
    "fnr1, fnr2, denom1, denom2, expected_p",
    [
        (0.2, 0.25, 100, 100, 0.3963327),
        (0.1, 0.15, 20, 35, np.nan),
        (0.3, 0.1, 200, 200, 2.4e-07),
    ],
)
def test_z_test_fnr_difference(fnr1, fnr2, denom1, denom2, expected_p):
    p_value = bias_detection.z_test_fnr_difference(fnr1, fnr2, denom1, denom2)
    assert np.isclose(p_value, expected_p, equal_nan=True)


@pytest.mark.parametrize(
    "group, sub1, sub2, fnr_percentage_difference, bias_type, split_name, flag, p, expected",
    [
        (
            "Gender",
            "Male",
            "Female",
            0.12,
            "non-overlapping confidence intervals",
            "train",
            "🔴 HIGH BIAS",
            0.005,
            {
                "group": "Gender",
                "subgroups": "Male vs Female",
                "fnr_percentage_difference": 0.12,
                "type": "non-overlapping confidence intervals with a p-value of 0.005",
                "split_name": "train",
                "flag": "🔴 HIGH BIAS",
                "p_value": 0.005,
            },
        ),
    ],
)
def test_generate_bias_flag(
    group,
    sub1,
    sub2,
    fnr_percentage_difference,
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
            fnr_percentage_difference,
            bias_type,
            split_name,
            flag,
            p,
        )
        == expected
    )


@pytest.mark.parametrize(
    "fnr_data, expected_sg1, expected_sg2, expected_flag",
    [
        # Case 1: Valid HIGH BIAS (diff > 0.15 and p ≤ 0.01)
        (
            [
                {
                    "group": "Race",
                    "subgroup": "GroupA",
                    "fnr": 0.30,
                    "size": 100,
                    "ci": (0.25, 0.35),
                    "number_of_positive_samples": 20,
                    "split_name": "validation",
                },
                {
                    "group": "Race",
                    "subgroup": "GroupB",
                    "fnr": 0.10,
                    "size": 100,
                    "ci": (0.05, 0.15),
                    "number_of_positive_samples": 20,
                    "split_name": "validation",
                },
            ],
            "GroupA",
            "GroupB",
            "🔴 HIGH BIAS",
        ),
        # Case 2: FNR diff > 0.1, but p > 0.01 -> LOW BIAS
        (
            [
                {
                    "group": "Race",
                    "subgroup": "GroupA",
                    "fnr": 0.25,
                    "size": 100,
                    "ci": (0.20, 0.30),
                    "number_of_positive_samples": 20,
                    "split_name": "validation",
                },
                {
                    "group": "Race",
                    "subgroup": "GroupB",
                    "fnr": 0.10,
                    "size": 100,
                    "ci": (0.08, 0.12),
                    "number_of_positive_samples": 20,
                    "split_name": "validation",
                },
            ],
            "GroupA",
            "GroupB",
            "🟡 LOW BIAS",
        ),
        # Case 3: FNR diff > 0.1, but p > 0.1 -> NO BIAS
        (
            [
                {
                    "group": "Race",
                    "subgroup": "GroupA",
                    "fnr": 0.25,
                    "size": 100,
                    "ci": (0.20, 0.30),
                    "number_of_positive_samples": 20,
                    "split_name": "validation",
                },
                {
                    "group": "Race",
                    "subgroup": "GroupB",
                    "fnr": 0.10,
                    "size": 100,
                    "ci": (0.08, 0.12),
                    "number_of_positive_samples": 20,
                    "split_name": "validation",
                },
            ],
            "GroupA",
            "GroupB",
            "🟢 NO BIAS",
        ),
    ],
)
def test_flag_bias(fnr_data, expected_sg1, expected_sg2, expected_flag):
    def mock_z_test_fnr_difference(fnr1, fnr2, n1, n2):
        if fnr1 == 0.30:
            return 0.005  # Case 1 -> High bias (p ≤ 0.01)
        elif fnr1 == 0.25 and fnr2 == 0.10:
            # Return different p-values based on test case
            return 0.03 if expected_flag == "🟡 LOW BIAS" else 0.15
        return 0.5

    def mock_check_ci_overlap(ci1, ci2):
        return False  # Assume no overlap in all test cases

    def mock_generate_bias_flag(
        group, sg1, sg2, fnr_diff, reason, split_name, flag, p_value
    ):
        return {
            "group": group,
            "sg1": sg1,
            "sg2": sg2,
            "flag": flag,
            "fnr_diff": fnr_diff,
            "p_value": p_value,
        }

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        bias_detection, "z_test_fnr_difference", mock_z_test_fnr_difference
    )
    monkeypatch.setattr(bias_detection, "check_ci_overlap", mock_check_ci_overlap)
    monkeypatch.setattr(bias_detection, "generate_bias_flag", mock_generate_bias_flag)

    result = bias_detection.flag_bias(
        fnr_data,
        high_bias_thresh=0.15,
        moderate_bias_thresh=0.1,
        low_bias_thresh=0.05,
        min_sample_ratio=0.1,
    )

    assert len(result) == 1
    flag = result[0]
    assert flag["sg1"] == expected_sg1
    assert flag["sg2"] == expected_sg2
    assert flag["flag"] == expected_flag

    monkeypatch.undo()


def test_aggregate_bias_scores_basic_case():
    flags = [
        {
            "split_name": "test",
            "flag": "🔴 HIGH BIAS",
            "fnr_percentage_difference": 0.6,
        },
        {
            "split_name": "test",
            "flag": "🟠 MODERATE BIAS",
            "fnr_percentage_difference": 0.4,
        },
        {"split_name": "test", "flag": "🟡 LOW BIAS", "fnr_percentage_difference": 0.2},
        {
            "split_name": "test",
            "flag": "⚪ INSUFFICIENT DATA",
            "fnr_percentage_difference": 0.5,
        },
        {"split_name": "test", "flag": "🟢 NO BIAS", "fnr_percentage_difference": 0.06},
        {"split_name": "val", "flag": "🔴 HIGH BIAS", "fnr_percentage_difference": 0.9},
    ]

    weights = bias_detection.FLAG_WEIGHTS

    scores = [
        0.6 * weights["🔴 HIGH BIAS"],
        0.4 * weights["🟠 MODERATE BIAS"],
        0.2 * weights["🟡 LOW BIAS"],
    ]
    raw = [0.6, 0.4, 0.2]

    expected = {
        "bias_score_sum": round(sum(scores), 4),
        "bias_score_mean": round(sum(scores) / 4, 4),  # 4 valid comparisons for test
        "bias_score_max": round(max(raw), 4),
        "num_bias_flags": 3,
        "num_valid_comparisons": 4,
    }

    result = bias_detection.aggregate_bias_scores(flags, split="test")
    assert result == expected


def test_aggregate_bias_scores_all_insufficient():
    flags = [
        {
            "split_name": "test",
            "flag": "⚪ INSUFFICIENT DATA",
            "fnr_percentage_difference": 0.3,
            "p_value": 0.5,
        },
        {
            "split_name": "test",
            "flag": "⚪ INSUFFICIENT DATA",
            "fnr_percentage_difference": 0.2,
            "p_value": 0.4,
        },
    ]
    result = bias_detection.aggregate_bias_scores(flags, split="test")

    expected = {
        "bias_score_sum": 0.0,
        "bias_score_mean": 0.0,
        "bias_score_max": 0.0,
        "num_bias_flags": 0,
        "num_valid_comparisons": 0,
    }

    assert result == expected


def test_aggregate_bias_scores_empty_input():
    result = bias_detection.aggregate_bias_scores([], split="test")
    assert result == {
        "bias_score_sum": 0.0,
        "bias_score_mean": 0.0,
        "bias_score_max": 0.0,
        "num_bias_flags": 0,
        "num_valid_comparisons": 0,
    }
