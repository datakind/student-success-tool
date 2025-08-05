import pytest
import pandas as pd
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


@mock.patch("student_success_tool.modeling.utils_h2o.log_h2o_experiment")
@mock.patch("student_success_tool.modeling.training_h2o.H2OAutoML")
@mock.patch("student_success_tool.modeling.training_h2o.h2o.H2OFrame")
@mock.patch("student_success_tool.modeling.training_h2o.correct_h2o_dtypes")
def test_run_h2o_automl_success(
    mock_correct, mock_h2o_frame, mock_automl, mock_log_experiment, sample_df
):
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
        target_name="Dropout Risk",
        checkpoint_name="ckpt_001",
        workspace_path="mlflow_experiments/",
    )

    assert aml == mock_automl_instance
    assert train == mock_frame
    assert valid == mock_frame
    assert test == mock_frame

    mock_automl_instance.train.assert_called_once()
    mock_log_experiment.assert_called_once_with(
        aml=mock_automl_instance,
        train=mock_frame,
        valid=mock_frame,
        test=mock_frame,
        institution_id="inst1",
        target_col="target",
        target_name="Dropout Risk",
        checkpoint_name="ckpt_001",
        workspace_path="mlflow_experiments/",
        client=mock.ANY,
    )


def test_run_h2o_automl_missing_logging_param(sample_df):
    with pytest.raises(ValueError, match="Missing logging parameters"):
        training_h2o.run_h2o_automl_classification(
            sample_df,
            target_col="target",
            primary_metric="AUC",
            institution_id="inst1",
            student_id_col="student_id",
            target_name=None,
            checkpoint_name="ckpt_001",
            workspace_path="mlflow_experiments/",
        )
