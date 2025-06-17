import mlflow.tracking
from unittest.mock import patch
import pandas as pd
import pytest

from student_success_tool.modeling import evaluation


@pytest.fixture
def patch_mlflow(monkeypatch):
    """Patch the mlflow.search_runs function."""

    def mock_search_runs():
        # This mock function does nothing and will be overridden by each test case
        pass

    monkeypatch.setattr(mlflow.tracking.MlflowClient, "search_runs", mock_search_runs)


def test_check_array_of_arrays_true():
    input_array = pd.Series([[1, 0, 1], [0, 1, 0]])
    assert evaluation._check_array_of_arrays(input_array)


def test_check_array_of_arrays_false():
    input_array = pd.Series([1, 0, 1])
    assert not evaluation._check_array_of_arrays(input_array)


@pytest.mark.parametrize(
    ["y_true", "risk_score", "q", "sensitivity"],
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
        evaluation.get_sensitivity_of_top_q_pctl_thresh(
            y_true, risk_score, q, pos_label="Y"
        )
        == sensitivity
    )


@pytest.mark.parametrize(
    ["data", "metric", "expected_order", "expected_columns"],
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
    result, _ = evaluation.compare_trained_models("dummy_id", metric)
    print(result["tags.model_type"].tolist())
    assert isinstance(result, pd.DataFrame), "The result should be a pandas DataFrame."
    assert result["tags.model_type"].tolist() == expected_order, (
        "Models are not sorted in ascending order based on the metric."
    )
    assert all(column in result.columns for column in expected_columns), (
        "DataFrame should contain specific columns."
    )


@pytest.fixture
def mock_runs_df():
    return pd.DataFrame({
        "run_id": ["r1", "r2", "r3"],
        "tags.mlflow.runName": ["run_1", "run_2", "run_3"],
        "metrics.test_roc_auc": [0.80, 0.60, 0.90],      # higher is better
        "metrics.test_recall_score": [0.70, 0.95, 0.60], # higher is better
        "metrics.val_log_loss": [0.25, 0.20, 0.30],      # lower is better
    })

@patch("your_module.mlflow.search_runs")
def test_get_top_runs_balanced(mock_search_runs, mock_runs_df):
    mock_search_runs.return_value = mock_runs_df

    top = evaluation.get_top_runs(
        experiment_id="dummy",
        optimization_metrics=["test_roc_auc", "test_recall_score", "val_log_loss"],
        topn_runs_included=1,
        debug=True,
    )

    # r1 is most balanced: decent AUC, recall, and log loss
    assert isinstance(top, dict)
    assert list(top.keys()) == ["run_1"]
    assert list(top.values()) == ["r1"]

@pytest.mark.parametrize(
    "metrics, expected_top",
    [
        (["test_recall_score"], "run_2"),       # r2 has best recall (0.95)
        (["val_log_loss"], "run_2"),            # r2 has lowest log loss (0.20)
        (["test_roc_auc"], "run_3"),            # r3 has best AUC (0.90)
        (["test_roc_auc", "val_log_loss"], "run_1"),  # r1 balances AUC and low loss
    ],
)
@patch("your_module.mlflow.search_runs")
def test_get_top_runs_parametrized(mock_search_runs, metrics, expected_top, mock_runs_df):
    mock_search_runs.return_value = mock_runs_df

    top = evaluation.get_top_runs(
        experiment_id="dummy",
        optimization_metrics=metrics,
        topn_runs_included=1,
        debug=False,
    )

    assert list(top.keys())[0] == expected_top