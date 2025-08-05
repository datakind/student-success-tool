import pytest
import pandas as pd
import numpy as np
import unittest.mock as mock
from student_success_tool.modeling import training_h2o


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "feature1": [1, 2, 3],
            "feature2": ["a", "b", "c"],
            "split": ["train", "validate", "test"],
            "target": [0, 1, 1],
            "student_id": [100, 101, 102],
        }
    )


@mock.patch("student_success_tool.modeling.training_h2o.H2OAutoML")
@mock.patch("student_success_tool.modeling.training_h2o.h2o.H2OFrame")
@mock.patch("student_success_tool.modeling.training_h2o.correct_h2o_dtypes")
def test_run_h2o_automl_success(mock_correct, mock_h2o_frame, mock_automl, sample_df):
    mock_frame = mock.MagicMock()
    mock_frame.columns = sample_df.columns.tolist()
    mock_frame.__getitem__.side_effect = lambda k: mock_frame

    mock_correct.return_value = mock_frame
    mock_h2o_frame.return_value = mock_frame
    mock_automl_instance = mock.MagicMock()
    mock_automl.return_value = mock_automl_instance
    mock_automl_instance.leader.model_id = "dummy_model"

    aml, train, valid, test = training_h2o.run_h2o_automl_classification(
        sample_df,
        target_col="target",
        primary_metric="AUC",
        institution_id="inst1",
        student_id_col="student_id",
    )

    assert aml == mock_automl_instance
    assert train == mock_frame
    assert valid == mock_frame
    assert test == mock_frame
    mock_automl_instance.train.assert_called_once()


def test_run_h2o_automl_missing_target(sample_df):
    with pytest.raises(ValueError):
        training_h2o.run_h2o_automl_classification(
            sample_df.drop(columns=["target"]),
            target_col="target",
            primary_metric="AUC",
            institution_id="inst1",
            student_id_col="student_id",
        )


def test_correct_h2o_dtypes_enum_conversion():
    df = pd.DataFrame({"feature": ["a", "b", "a"]})
    h2o_df = mock.MagicMock()
    h2o_df.columns = ["feature"]
    h2o_df.types = {"feature": "real"}

    result = training_h2o.correct_h2o_dtypes(h2o_df, df)
    assert result == h2o_df


@mock.patch("student_success_tool.modeling.training_h2o.mlflow.log_artifact")
@mock.patch("student_success_tool.modeling.training_h2o.mlflow.start_run")
@mock.patch("student_success_tool.modeling.training_h2o.mlflow.active_run")
@mock.patch("student_success_tool.modeling.training_h2o.evaluate_and_log_model")
def test_log_h2o_experiment_logs_metrics(
    mock_eval_log,
    mock_active_run,
    mock_start_run,
    mock_log_artifact,
):
    mock_aml = mock.MagicMock()
    mock_aml.leaderboard.as_data_frame.return_value = pd.DataFrame(
        {"model_id": ["model1"]}
    )
    mock_aml.leader.model_id = "model1"

    client_mock = mock.MagicMock()
    experiment_mock = mock.MagicMock()
    experiment_mock.experiment_id = "exp123"
    client_mock.get_experiment_by_name.return_value = experiment_mock

    # Evaluation mock
    mock_eval_log.return_value = {"accuracy": 0.9, "model_id": "model1"}

    # Active run and start_run mocks
    mock_active_run.return_value.info.run_id = "parent-run-id"
    mock_start_run.return_value.__enter__.return_value.info.run_id = "parent-run-id"

    experiment_id, results_df = training_h2o.log_h2o_experiment(
        aml=mock_aml,
        train=mock.MagicMock(),
        valid=mock.MagicMock(),
        test=mock.MagicMock(),
        institution_id="inst1",
        target_col="target",
        target_name="Outcome",
        checkpoint_name="CP",
        workspace_path="/workspace",
        client=client_mock,
    )

    assert experiment_id == "exp123"
    assert not results_df.empty
    assert "accuracy" in results_df.columns
    assert results_df["model_id"].iloc[0] == "model1"


@mock.patch("student_success_tool.modeling.training_h2o.h2o.save_model")
@mock.patch("student_success_tool.modeling.training_h2o.mlflow.active_run")
@mock.patch("student_success_tool.modeling.training_h2o.mlflow.start_run")
@mock.patch("student_success_tool.modeling.training_h2o.h2o.get_model")
@mock.patch("student_success_tool.modeling.training_h2o.evaluation")
def test_evaluate_and_log_model_success(
    mock_eval,
    mock_get_model,
    mock_start_run,
    mock_active_run,
    mock_save_model,
):
    model_mock = mock.MagicMock()
    model_mock.predict.return_value.col_names = ["p0", "p1"]
    model_mock.predict.return_value.__getitem__.return_value.as_data_frame.return_value.values.flatten.return_value = np.array(
        [0.6, 0.7, 0.8]
    )

    mock_get_model.return_value = model_mock

    mock_eval.get_metrics_near_threshold_all_splits.return_value = {"accuracy": 0.91}
    mock_eval.generate_all_classification_plots.return_value = None

    mock_start_run.return_value.__enter__.return_value.info.run_id = "run-xyz"

    result = training_h2o.evaluate_and_log_model(
        aml=mock.MagicMock(),
        model_id="model1",
        train=mock.MagicMock(),
        valid=mock.MagicMock(),
        test=mock.MagicMock(),
        client=mock.MagicMock(),
    )

    assert isinstance(result, dict)
    assert "mlflow_run_id" in result
    mock_save_model.assert_called_once()


@mock.patch("student_success_tool.modeling.training_h2o.mlflow.set_experiment")
def test_set_or_create_experiment_new(mock_set_experiment):
    mock_client = mock.MagicMock()
    mock_client.get_experiment_by_name.return_value = None
    mock_client.create_experiment.return_value = "exp-123"

    exp_id = training_h2o.set_or_create_experiment(
        "/workspace", "inst1", "target", "chkpt1", client=mock_client
    )
    assert exp_id == "exp-123"


@mock.patch("student_success_tool.modeling.training_h2o.mlflow.set_experiment")
def test_set_or_create_experiment_existing(mock_set_experiment):
    mock_client = mock.MagicMock()
    mock_client.get_experiment_by_name.return_value = mock.MagicMock(
        experiment_id="exp-456"
    )

    exp_id = training_h2o.set_or_create_experiment(
        "/workspace", "inst1", "target", "chkpt1", client=mock_client
    )
    assert exp_id == "exp-456"


@mock.patch("student_success_tool.modeling.training_h2o.mlflow.log_artifact")
@mock.patch("student_success_tool.modeling.training_h2o.mlflow.log_param")
@mock.patch("student_success_tool.modeling.training_h2o.mlflow.log_metric")
@mock.patch("student_success_tool.modeling.training_h2o.mlflow.start_run")
def test_log_h2o_experiment_summary_basic(
    mock_start_run,
    mock_log_metric,
    mock_log_param,
    mock_log_artifact,
):
    # Patch the mlflow.start_run context manager
    mock_run = mock.MagicMock()
    mock_run.__enter__.return_value = mock_run
    mock_run.__exit__.return_value = None
    mock_start_run.return_value = mock_run

    # Mock aml with leader.model_id
    aml_mock = mock.Mock()
    aml_mock.leader.model_id = "best_model"

    # Mock leaderboard
    leaderboard_df = pd.DataFrame(
        {"model_id": ["model_1", "model_2"], "auc": [0.9, 0.85]}
    )

    # Mock H2OFrame train
    train_mock = mock.MagicMock()
    train_mock.columns = ["feature_1", "feature_2", "target"]
    train_mock.types = {"feature_1": "real", "feature_2": "int", "target": "enum"}
    train_df_mock = pd.DataFrame(
        {"feature_1": [0.1, 0.2], "feature_2": [1, 2], "target": [0, 1]}
    )
    train_mock.as_data_frame.return_value = train_df_mock

    table_df_mock = pd.DataFrame({"target": [0, 1], "Count": [1, 1]})
    target_col_mock = mock.Mock()
    target_col_mock.table.return_value.as_data_frame.return_value = table_df_mock
    train_mock.__getitem__.return_value = target_col_mock

    # Call the function
    training_h2o.log_h2o_experiment_summary(
        aml=aml_mock,
        leaderboard_df=leaderboard_df,
        train=train_mock,
        target_col="target",
    )

    # Assertions
    mock_start_run.assert_called_once()
    mock_log_metric.assert_called_once_with("num_models_trained", 2)
    mock_log_param.assert_called_once_with("best_model_id", "best_model")
    assert (
        mock_log_artifact.call_count == 5
    )  # leaderboard + features + train.csv + target dist + schema
