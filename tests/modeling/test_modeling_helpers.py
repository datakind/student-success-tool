from unittest.mock import MagicMock, patch

import mlflow.tracking
import numpy as np
import pandas as pd
import pytest

from student_success_tool.modeling.modeling_helpers import (
    check_array_of_arrays,
    compare_trained_models,
    drop_collinear_features_iteratively,
    drop_incomplete_features,
    drop_low_variance_features,
    get_sensitivity_of_top_q_pctl_thresh,
    run_automl_classification,
)


@pytest.mark.parametrize(
    "threshold,expected_columns",
    [
        (0.4, ["complete_feature"]),
        (0.5, ["complete_feature", "kinda_incomplete_feature"]),
    ],
)
def test_drop_incomplete_features(threshold, expected_columns):
    test_features = pd.DataFrame(
        {
            "complete_feature": [1, 2, 3, 4, 5],
            "kinda_incomplete_feature": [np.nan, np.nan, 3, 4, 5],
            "incomplete_feature": [np.nan, 2, np.nan, np.nan, np.nan],
        }
    )
    returned_df = drop_incomplete_features(test_features, threshold)
    assert set(returned_df.columns.values) == set(expected_columns)
    assert returned_df.shape[0] == test_features.shape[0]


def test_drop_low_variance_features():
    low_variance_features = pd.DataFrame(
        {"has_variance": [1, 2, 3, 4, 5], "no_variance": [1, 1, 1, 1, 1]}
    )
    returned_df = drop_low_variance_features(low_variance_features, 0.0)
    assert set(returned_df.columns.values) == {"has_variance"}
    assert returned_df.shape[0] == low_variance_features.shape[0]


def test_drop_low_variance_features_no_dropped_columns_incl_categorical():
    low_variance_features = pd.DataFrame(
        {
            "has_variance": [1, 2, 3, 4, 5],
            "categorical": ["cat", "eg", "or", "ic", "al"],
        }
    )
    returned_df = drop_low_variance_features(low_variance_features, 0.0)
    assert set(returned_df.columns.values) == set(low_variance_features.columns.values)
    assert returned_df.shape[0] == low_variance_features.shape[0]


def test_drop_collinear_features_iteratively():
    related_features = pd.DataFrame(
        {"A": [1, 2, 3, 4, 5], "B": [6, 7, 8, 9, 10], "C": [6.5, 1.5, 4, 3.5, 2]}
    )
    returned_df = drop_collinear_features_iteratively(
        related_features, force_include_cols=[]
    )
    assert set(returned_df.columns.values) == {"A", "C"}
    assert returned_df.shape[0] == related_features.shape[0]


def test_drop_collinear_features_iteratively_force_include():
    related_features = pd.DataFrame(
        {"A": [1, 2, 3, 4, 5], "B": [6, 7, 8, 9, 10], "C": [6.5, 1.5, 4, 3.5, 2]}
    )
    returned_df = drop_collinear_features_iteratively(
        related_features, force_include_cols=["B"]
    )
    assert set(returned_df.columns.values) == {"B", "C"}
    assert returned_df.shape[0] == related_features.shape[0]


def test_drop_collinear_features_iteratively_fails_with_one_feature_left():
    related_features = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [6, 7, 8, 9, 10]})
    with pytest.raises(Exception):
        drop_collinear_features_iteratively(related_features)


def test_run_automl_classification_uses_correct_args_and_format():
    mymodule = MagicMock()

    train_df = pd.DataFrame(
        {"id": [1, 2, 3], "semester_start": [1, 1, 2], "didntgrad": [1, 0, 1]}
    )
    outcome_col = "didntgrad"
    automl_metric = "log_loss"
    student_id_col = "id"
    input_kwargs = {"time_col": "semester_start", "timeout_minutes": 20}

    with patch.dict("sys.modules", databricks=mymodule):
        run_automl_classification(
            institution_id="test_inst",
            job_run_id="test",
            train_df=train_df,
            outcome_col=outcome_col,
            optimization_metric=automl_metric,
            student_id_col=student_id_col,
            **input_kwargs,
        )
        _, kwargs = mymodule.automl.classify.call_args
        pd.testing.assert_frame_equal(kwargs.get("dataset"), train_df)
        assert kwargs.get("target_col") == outcome_col
        assert kwargs.get("primary_metric") == automl_metric
        # particularly interested in making sure this was called as a list
        assert kwargs.get("exclude_cols") == [student_id_col]
        assert kwargs.get("time_col") == input_kwargs["time_col"]
        assert kwargs.get("timeout_minutes") == input_kwargs["timeout_minutes"]
        assert kwargs.get("pos_label") == True


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
    monkeypatch.setattr(mlflow, "search_runs", lambda x: pd.DataFrame(data))
    result, _ = compare_trained_models("dummy_id", metric)
    print(result["tags.model_type"].tolist())
    assert isinstance(result, pd.DataFrame), "The result should be a pandas DataFrame."
    assert (
        result["tags.model_type"].tolist() == expected_order
    ), "Models are not sorted in ascending order based on the metric."
    assert all(
        column in result.columns for column in expected_columns
    ), "DataFrame should contain specific columns."
