import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
import unittest.mock as mock


from student_success_tool.modeling.h2o_modeling import training


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


def make_mock_frame(columns):
    mock_frame = mock.MagicMock()
    mock_frame.columns = columns

    col_mocks = {}

    for col in columns:
        call_count = {"val": 0}
        col_mock = mock.MagicMock(name=f"{col}_col_mock")

        def make_sum_side_effect():
            def _side_effect():
                return 0

            return _side_effect

        isna_mock = mock.MagicMock()
        isna_mock.sum.side_effect = make_sum_side_effect()
        isna_mock.ifelse.return_value = col_mock
        col_mock.isna.return_value = isna_mock

        col_mock.isfactor.return_value = False
        col_mock.levels.return_value = [["A", "B", "C"]]
        col_mock.asfactor.return_value = col_mock

        col_mocks[col] = col_mock

    mock_frame.__getitem__.side_effect = lambda col: (
        col_mocks[col] if isinstance(col, str) else mock_frame
    )
    mock_frame.__setitem__.side_effect = lambda col, val: None

    return mock_frame


@mock.patch(
    "student_success_tool.modeling.h2o_modeling.training.utils.set_or_create_experiment"
)
@mock.patch("student_success_tool.modeling.h2o_modeling.utils.log_h2o_experiment")
@mock.patch("student_success_tool.modeling.h2o_modeling.training.H2OAutoML")
@mock.patch("student_success_tool.modeling.h2o_modeling.training.h2o.H2OFrame")
@mock.patch("student_success_tool.modeling.h2o_modeling.training.correct_h2o_dtypes")
@mock.patch(
    "student_success_tool.modeling.h2o_modeling.training.imputation.SklearnImputerWrapper"
)
def test_run_h2o_automl_success(
    mock_imputer_cls,
    mock_correct,
    mock_h2o_frame,
    mock_automl,
    mock_log_experiment,
    mock_set_experiment,
    sample_df,
):
    columns = sample_df.columns.tolist()

    # Mock imputer
    mock_imputer = mock.MagicMock()
    mock_imputer.fit.return_value = None
    mock_imputer.transform.side_effect = lambda df: pd.DataFrame(
        df.values, columns=df.columns
    )
    mock_imputer_cls.return_value = mock_imputer

    # Mock H2O frame + dtype correction
    train_frame = make_mock_frame(columns)
    valid_frame = make_mock_frame(columns)
    test_frame = make_mock_frame(columns)
    mock_h2o_frame.return_value = make_mock_frame(columns)
    mock_correct.side_effect = [train_frame, valid_frame, test_frame]

    # Mock AutoML
    mock_automl_instance = mock.MagicMock()
    mock_automl_instance.leader.model_id = "dummy_model"
    mock_automl.return_value = mock_automl_instance

    # Mock MLflow
    mock_log_experiment.return_value = ("exp-123", pd.DataFrame())
    mock_set_experiment.return_value = "exp-123"

    experiment_id, aml, train, valid, test = training.run_h2o_automl_classification(
        sample_df,
        target_col="target",
        primary_metric="AUC",
        institution_id="inst1",
        student_id_col="student_id",
        target_name="Dropout Risk",
        checkpoint_name="ckpt_001",
        workspace_path="mlflow_experiments/",
    )

    assert experiment_id == "exp-123"
    assert aml == mock_automl_instance
    assert train.columns == train_frame.columns
    assert valid.columns == valid_frame.columns
    assert test.columns == test_frame.columns

    raw_model_features = [
        c for c in sample_df.columns if c not in ["target", "split", "student_id"]
    ]
    expected_df = sample_df.loc[sample_df["split"] == "train", raw_model_features]
    actual_df = mock_imputer.fit.call_args[0][0]
    assert_frame_equal(actual_df, expected_df)


@mock.patch("student_success_tool.modeling.h2o_modeling.training.h2o.H2OFrame")
def test_run_h2o_automl_missing_logging_param(mock_h2oframe, sample_df):
    fake_h2o_frame = mock.MagicMock()
    fake_h2o_frame.columns = sample_df.columns.tolist()
    mock_h2oframe.return_value = fake_h2o_frame

    with pytest.raises(ValueError, match="Missing logging parameters: target_name"):
        training.run_h2o_automl_classification(
            sample_df,
            target_col="target",
            primary_metric="AUC",
            institution_id="inst1",
            student_id_col="student_id",
            # target_name is missing
            checkpoint_name="ckpt_001",
            workspace_path="mlflow_experiments/",
        )
