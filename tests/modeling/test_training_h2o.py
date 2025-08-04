import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, ANY, MagicMock
from mlflow.tracking import MlflowClient
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


@patch("student_success_tool.modeling.training_h2o.H2OAutoML")
@patch("student_success_tool.modeling.training_h2o.h2o.H2OFrame")
@patch("student_success_tool.modeling.training_h2o.correct_h2o_dtypes")
def test_run_h2o_automl_success(mock_correct, mock_h2o_frame, mock_automl, sample_df):
    mock_frame = MagicMock()
    mock_frame.columns = sample_df.columns.tolist()
    mock_frame.__getitem__.side_effect = lambda k: mock_frame

    mock_correct.return_value = mock_frame
    mock_h2o_frame.return_value = mock_frame
    mock_automl_instance = MagicMock()
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
    h2o_df = MagicMock()
    h2o_df.columns = ["feature"]
    h2o_df.types = {"feature": "real"}

    result = training_h2o.correct_h2o_dtypes(h2o_df, df)
    assert result == h2o_df


@patch("student_success_tool.modeling.training_h2o.evaluate_and_log_model")
def test_log_h2o_experiment_logs_metrics(mock_eval_log):
    mock_aml = MagicMock()
    mock_aml.leaderboard.as_data_frame.return_value = pd.DataFrame(
        {"model_id": ["model1"]}
    )
    client_mock = MagicMock()
    experiment_mock = MagicMock()
    experiment_mock.experiment_id = "exp123"
    client_mock.get_experiment_by_name.return_value = experiment_mock
    mock_eval_log.return_value = {"accuracy": 0.9, "model_id": "model1"}

    experiment_id, results_df = training_h2o.log_h2o_experiment(
        aml=mock_aml,
        train=MagicMock(),
        valid=MagicMock(),
        test=MagicMock(),
        institution_id="inst1",
        target_col="target",
        target_name="Outcome",
        checkpoint_name="CP",
        workspace_path="/workspace",
        client=client_mock,
    )
    assert isinstance(experiment_id, str)
    assert len(results_df) == 1
    assert "accuracy" in results_df.columns
    mock_eval_log.assert_called_once_with(
        mock_aml, "model1", ANY, ANY, ANY, 0.5, client_mock
    )


@patch("student_success_tool.modeling.training_h2o.h2o.save_model")
@patch("student_success_tool.modeling.training_h2o.mlflow.start_run")
@patch("student_success_tool.modeling.training_h2o.h2o.get_model")
@patch("student_success_tool.modeling.training_h2o.evaluation")
def test_evaluate_and_log_model_success(
    mock_eval, mock_get_model, mock_start_run, mock_save_model
):
    model_mock = MagicMock()
    model_mock.predict.return_value.col_names = ["p0", "p1"]
    model_mock.predict.return_value.__getitem__.return_value.as_data_frame.return_value.values.flatten.return_value = np.array(
        [0.6, 0.7, 0.8]
    )

    mock_get_model.return_value = model_mock

    mock_eval.get_metrics_near_threshold_all_splits.return_value = {"accuracy": 0.91}
    mock_eval.generate_all_classification_plots.return_value = None

    mock_start_run.return_value.__enter__.return_value.info.run_id = "run-xyz"

    result = training_h2o.evaluate_and_log_model(
        aml=MagicMock(),
        model_id="model1",
        train=MagicMock(),
        valid=MagicMock(),
        test=MagicMock(),
        client=MagicMock(),
    )

    assert isinstance(result, dict)
    assert "mlflow_run_id" in result


@patch("student_success_tool.modeling.training_h2o.mlflow.set_experiment")
def test_set_or_create_experiment_new(mock_set_experiment):
    mock_client = MagicMock()
    mock_client.get_experiment_by_name.return_value = None
    mock_client.create_experiment.return_value = "exp-123"

    exp_id = training_h2o.set_or_create_experiment(
        "/workspace", "inst1", "target", "chkpt1", client=mock_client
    )
    assert exp_id == "exp-123"


@patch("student_success_tool.modeling.training_h2o.mlflow.set_experiment")
def test_set_or_create_experiment_existing(mock_set_experiment):
    mock_client = MagicMock()
    mock_client.get_experiment_by_name.return_value = MagicMock(experiment_id="exp-456")

    exp_id = training_h2o.set_or_create_experiment(
        "/workspace", "inst1", "target", "chkpt1", client=mock_client
    )
    assert exp_id == "exp-456"
