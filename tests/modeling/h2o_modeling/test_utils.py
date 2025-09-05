import os
import unittest.mock as mock
import pandas as pd
import pytest

from student_success_tool.modeling.h2o_modeling import utils


@mock.patch("student_success_tool.modeling.h2o_modeling.utils.mlflow.log_param")
@mock.patch("student_success_tool.modeling.h2o_modeling.utils.mlflow.log_metric")
@mock.patch("student_success_tool.modeling.h2o_modeling.utils.mlflow.log_artifact")
@mock.patch("student_success_tool.modeling.h2o_modeling.utils.mlflow.start_run")
@mock.patch("student_success_tool.modeling.h2o_modeling.utils.mlflow.active_run")
@mock.patch("student_success_tool.modeling.h2o_modeling.utils.log_h2o_model")
def test_log_h2o_experiment_logs_metrics(
    mock_eval_log,
    mock_active_run,
    mock_start_run,
    mock_log_artifact,
    mock_log_metric,
    mock_log_param,
):
    mock_aml = mock.MagicMock()
    mock_aml.leaderboard.as_data_frame.return_value = pd.DataFrame(
        {"model_id": ["model1"]}
    )
    mock_aml.leader.model_id = "model1"
    mock_aml.sort_metric = "logloss"

    # Evaluation mock
    mock_eval_log.return_value = {"accuracy": 0.9, "model_id": "model1"}

    # Active run and start_run mocks
    mock_active_run.return_value.info.run_id = "parent-run-id"
    mock_start_run.return_value.__enter__.return_value.info.run_id = "parent-run-id"

    train_mock = mock.MagicMock()
    train_mock.as_data_frame.return_value = pd.DataFrame({"target": [0, 1]})

    valid_mock = mock.MagicMock()
    valid_mock.as_data_frame.return_value = pd.DataFrame({"target": [0, 1]})

    test_mock = mock.MagicMock()
    test_mock.as_data_frame.return_value = pd.DataFrame({"target": [0, 1]})

    results_df = utils.log_h2o_experiment(
        aml=mock_aml,
        train=train_mock,
        valid=valid_mock,
        test=test_mock,
        target_col="target",
        experiment_id="exp123",
    )

    assert not results_df.empty
    assert "accuracy" in results_df.columns
    assert results_df["model_id"].iloc[0] == "model1"


@mock.patch("student_success_tool.modeling.h2o_modeling.utils.mlflow.set_experiment")
def test_set_or_create_experiment_new(mock_set_experiment):
    mock_client = mock.MagicMock()
    mock_client.get_experiment_by_name.return_value = None
    mock_client.create_experiment.return_value = "exp-123"

    exp_id = utils.set_or_create_experiment(
        "/workspace", "inst1", "target", "chkpt1", client=mock_client
    )
    assert exp_id == "exp-123"


@mock.patch("student_success_tool.modeling.h2o_modeling.utils.mlflow.set_experiment")
def test_set_or_create_experiment_existing(mock_set_experiment):
    mock_client = mock.MagicMock()
    mock_client.get_experiment_by_name.return_value = mock.MagicMock(
        experiment_id="exp-456"
    )

    exp_id = utils.set_or_create_experiment(
        "/workspace", "inst1", "target", "chkpt1", client=mock_client
    )
    assert exp_id == "exp-456"


@mock.patch("student_success_tool.modeling.h2o_modeling.utils.mlflow.log_artifact")
@mock.patch("student_success_tool.modeling.h2o_modeling.utils.mlflow.log_param")
@mock.patch("student_success_tool.modeling.h2o_modeling.utils.mlflow.log_metric")
@mock.patch("student_success_tool.modeling.h2o_modeling.utils.mlflow.start_run")
def test_log_h2o_experiment_summary_basic(
    mock_start_run,
    mock_log_metric,
    mock_log_param,
    mock_log_artifact,
):
    mock_run = mock.MagicMock()
    mock_run.__enter__.return_value = mock_run
    mock_start_run.return_value = mock_run

    # Mock AutoML
    aml_mock = mock.Mock()
    aml_mock.leader.model_id = "best_model"
    leaderboard_df = pd.DataFrame(
        {"model_id": ["model_1", "model_2"], "auc": [0.9, 0.85]}
    )

    # Train H2OFrame
    train_mock = mock.MagicMock()
    train_mock.columns = ["feature_1", "feature_2", "target"]
    train_mock.types = {"feature_1": "real", "feature_2": "int", "target": "enum"}
    train_df = pd.DataFrame(
        {"feature_1": [0.1, 0.2], "feature_2": [1, 2], "target": [0, 1]}
    )
    train_mock.as_data_frame.return_value = train_df

    # Add valid/test mocks
    valid_mock = mock.MagicMock()
    valid_mock.as_data_frame.return_value = train_df
    test_mock = mock.MagicMock()
    test_mock.as_data_frame.return_value = train_df

    # Target distribution
    target_col_mock = mock.Mock()
    target_col_mock.table.return_value.as_data_frame.return_value = pd.DataFrame(
        {"target": [0, 1], "Count": [1, 1]}
    )
    train_mock.__getitem__.return_value = target_col_mock

    utils.log_h2o_experiment_summary(
        aml=aml_mock,
        leaderboard_df=leaderboard_df,
        train=train_mock,
        valid=valid_mock,
        test=test_mock,
        target_col="target",
    )

    mock_start_run.assert_called_once()
    mock_log_metric.assert_called_once_with("num_models_trained", 2)
    mock_log_param.assert_called_once_with("best_model_id", "best_model")
    assert mock_log_artifact.call_count == 5


@mock.patch("student_success_tool.modeling.h2o_modeling.utils.h2o.save_model")
@mock.patch(
    "student_success_tool.modeling.h2o_modeling.utils.mlflow.log_artifact"
)  # single file
@mock.patch(
    "student_success_tool.modeling.h2o_modeling.utils.mlflow.log_text"
)  # MLmodel yaml
@mock.patch(
    "student_success_tool.modeling.h2o_modeling.utils.mlflow.log_metrics"
)  # batched metrics
@mock.patch("student_success_tool.modeling.h2o_modeling.utils.mlflow.log_param")
@mock.patch("student_success_tool.modeling.h2o_modeling.utils.mlflow.active_run")
@mock.patch("student_success_tool.modeling.h2o_modeling.utils.mlflow.start_run")
@mock.patch("student_success_tool.modeling.h2o_modeling.utils.evaluation")
@mock.patch("student_success_tool.modeling.h2o_modeling.utils.h2o.get_model")
@mock.patch("student_success_tool.modeling.h2o_modeling.utils.infer_signature")
def test_log_h2o_model_basic(
    mock_infer_signature,
    mock_get_model,
    mock_eval,
    mock_start_run,
    mock_active_run,
    mock_log_param,
    mock_log_metrics,  # batched
    mock_log_text,
    mock_log_artifact,
    mock_save_model,
):
    import pandas as pd
    from mlflow.models import infer_signature as real_infer_signature

    # real signature object (not a MagicMock)
    mock_infer_signature.return_value = real_infer_signature(
        pd.DataFrame({"x": [1.0, 2.0, 3.0]}),
        pd.DataFrame({"y": [0.0, 1.0, 0.0]}),
    )

    # ---- Mock H2O model
    mock_model = mock.MagicMock()
    mock_get_model.return_value = mock_model
    mock_save_model.return_value = "/tmp/h2o_models/m1"

    # predictions
    preds = mock.MagicMock()
    preds.col_names = ["p0", "p1"]
    prob_frame = mock.MagicMock()
    prob_frame.as_data_frame.return_value = pd.DataFrame({"p1": [0.8, 0.2]})
    preds.__getitem__.return_value = prob_frame
    mock_model.predict.return_value = preds

    # eval metrics
    mock_eval.get_metrics_near_threshold_all_splits.return_value = {
        "validate_logloss": 0.3
    }

    # H2OFrame → pandas for target
    target_frame = mock.MagicMock()
    target_frame.as_data_frame.return_value = pd.DataFrame({"target": [0, 1]})
    frame_mock = mock.MagicMock()
    frame_mock.__getitem__.return_value = target_frame

    # MLflow run
    mock_run = mock.MagicMock()
    mock_run.__enter__.return_value = mock_run
    mock_start_run.return_value = mock_run
    mock_active_run.return_value.info.run_id = "run-123"

    def fake_save_model(model, path, force=True):
        fname = os.path.join(path, "dummy_model")
        with open(fname, "w") as f:
            f.write("fake model")
        return fname

    mock_save_model.side_effect = fake_save_model

    # ---- Execute
    result = utils.log_h2o_model(
        aml=mock.MagicMock(),
        model_id="m1",
        train=frame_mock,
        valid=frame_mock,
        test=frame_mock,
        target_col="target",
    )

    # ---- Assertions
    assert result is not None
    assert "validate_logloss" in result
    assert result["mlflow_run_id"] == "run-123"

    # model_id param
    mock_log_param.assert_any_call("model_id", "m1")

    # batched metrics contain our value
    assert mock_log_metrics.called
    logged_metrics = mock_log_metrics.call_args[0][0]  # dict
    assert logged_metrics.get("validate_logloss") == 0.3

    # MLmodel + model.h2o were logged
    assert mock_log_text.called  # MLmodel yaml
    assert mock_log_artifact.called  # model/model.h2o
    mock_save_model.assert_called_once()


def test_correct_h2o_dtypes_converts_object_column():
    # Fake pandas DataFrame
    df = pd.DataFrame(
        {
            "cat": ["a", "b", "c"],
            "num": [1, 2, 3],
        }
    )

    # Fake H2OFrame
    fake_h2o = mock.MagicMock()
    fake_h2o.columns = ["cat", "num"]
    fake_h2o.types = {"cat": "int", "num": "real"}  # mis-inferred
    fake_h2o.__getitem__.side_effect = lambda c: fake_h2o  # allow hf[col]
    fake_h2o.asfactor.return_value = "converted"

    # Run
    result = utils.correct_h2o_dtypes(fake_h2o, df)

    # Assertions
    fake_h2o.__getitem__.assert_any_call("cat")
    assert result == fake_h2o
    fake_h2o.asfactor.assert_called_once()


@mock.patch(
    "student_success_tool.modeling.h2o_modeling.utils.correct_h2o_dtypes",
    return_value="corrected",
)
@mock.patch(
    "student_success_tool.modeling.h2o_modeling.utils.h2o.H2OFrame", return_value="hf"
)
def test_to_h2o_from_pandas_dataframe(mock_h2oframe, mock_correct):
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = utils._to_h2o(df)

    mock_h2oframe.assert_called_once_with(df)
    mock_correct.assert_called_once_with("hf", df, force_enum_cols=None)
    assert result == "corrected"


def test_to_pandas_from_h2oframe():
    class FakeH2OFrame:
        def as_data_frame(self, **kwargs):
            return pd.DataFrame({"a": [1, 2]})

    fake_hf = FakeH2OFrame()
    result = utils._to_pandas(fake_hf)

    assert isinstance(result, pd.DataFrame)


def test_to_pandas_from_two_dim_table():
    class FakeTwoDimTable:
        def as_data_frame(self):
            return pd.DataFrame({"metric": [1]})

    fake_tbl = FakeTwoDimTable()
    result = utils._to_pandas(fake_tbl)

    assert isinstance(result, pd.DataFrame)


def test_to_pandas_from_generic_object():
    class FakeObj:
        def as_data_frame(self):
            return pd.DataFrame({"x": [42]})

    result = utils._to_pandas(FakeObj())
    assert list(result.columns) == ["x"]


def test_to_pandas_unsupported_type():
    with pytest.raises(TypeError):
        utils._to_pandas(123)  # int is not supported
