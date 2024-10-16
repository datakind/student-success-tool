import mlflow.tracking
import pandas as pd
import pytest

from student_success_tool.modeling.evaluation import (
    check_array_of_arrays,
    compare_trained_models,
    get_sensitivity_of_top_q_pctl_thresh,
)


def test_check_array_of_arrays_true():
    input_array = pd.Series([[1, 0, 1], [0, 1, 0]])
    assert check_array_of_arrays(input_array)


def test_check_array_of_arrays_false():
    input_array = pd.Series([1, 0, 1])
    assert not check_array_of_arrays(input_array)


@pytest.mark.parametrize(
    "y_true,risk_score,q,sensitivity",
    [
        (["Y", "Y", "Y", "N", "N"], [0.12, 0.32, 0.98, 0.48, 0.87], 0.99, 1 / 3),
        (
            ["N", "N", "N", "N", "Y", "N", "N", "N", "N", "Y"],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
            0.9,
            0.5,
        ),
    ],
)
def test_get_sensitivity_of_top_q_pctl_thresh(y_true, risk_score, q, sensitivity):
    assert (
        get_sensitivity_of_top_q_pctl_thresh(y_true, risk_score, q, pos_label="Y")
        == sensitivity
    )


@pytest.fixture
def patch_mlflow(monkeypatch):
    """Patch the mlflow.search_runs function."""

    def mock_search_runs():
        # This mock function does nothing and will be overridden by each test case
        pass

    monkeypatch.setattr(mlflow.tracking.MlflowClient, "search_runs", mock_search_runs)


@pytest.mark.parametrize(
    "data, metric, expected_order, expected_columns",
    [
        (
            {
                "tags.model_type": ["Model A", "Model B", "Model C"],
                "metrics.val_recall_score": [0.92, 0.88, 0.93],
            },
            "recall",
            ["Model C", "Model A", "Model B"],
            ["tags.model_type", "metrics.val_recall_score"],
        ),
        (
            {
                "tags.model_type": ["Model D", "Model E", "Model F"],
                "metrics.val_f1_score": [0.80, 0.78, 0.81],
            },
            "f1",
            ["Model F", "Model D", "Model E"],
            ["tags.model_type", "metrics.val_f1_score"],
        ),
        (
            {
                "tags.model_type": ["Model G", "Model H", "Model I"],
                "metrics.val_log_loss": [0.4521, 0.3501, 0.5502],
            },
            "log_loss",
            ["Model H", "Model G", "Model I"],
            ["tags.model_type", "metrics.val_log_loss"],
        ),
    ],
)
def test_compare_trained_models(
    data, metric, expected_order, expected_columns, patch_mlflow, monkeypatch
):
    def _search_runs_patch(experiment_ids, output_format):
        return pd.DataFrame(data)

    monkeypatch.setattr(mlflow, "search_runs", _search_runs_patch)
    result, _ = compare_trained_models("dummy_id", metric)
    print(result["tags.model_type"].tolist())
    assert isinstance(result, pd.DataFrame), "The result should be a pandas DataFrame."
    assert (
        result["tags.model_type"].tolist() == expected_order
    ), "Models are not sorted in ascending order based on the metric."
    assert all(
        column in result.columns for column in expected_columns
    ), "DataFrame should contain specific columns."