import pytest
import pandas as pd
from unittest import mock
import json
import os

from student_success_tool.modeling import imputation_h2o as imputation


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "age": [25, 30, None, 45],
            "income": [50000, None, 70000, 60000],
            "gender": ["M", "F", "F", None],
            "student": [True, False, None, True],
        }
    )


def test_strategy_map(sample_df):
    wrapper = imputation.H2OImputerWrapper()
    strat_map = wrapper._assign_strategies(sample_df)

    assert strat_map["age"]["strategy"] in ["mean", "median"]
    assert strat_map["income"]["strategy"] == "mean"
    assert strat_map["gender"]["strategy"] == "mode"
    assert strat_map["student"]["strategy"] == "mode"

    assert isinstance(strat_map["age"]["value"], (int, float))
    assert isinstance(strat_map["gender"]["value"], str)

    assert strat_map["age"]["value"] == 30.0
    assert strat_map["income"]["value"] == 60000.0
    assert strat_map["gender"]["value"] == "F"
    assert strat_map["student"]["value"] is True


def test_fit(sample_df):
    wrapper = imputation.H2OImputerWrapper()

    def make_mock_frame():
        col_mocks = {}
        frame_mock = mock.MagicMock()

        for col in ["age", "income", "gender", "student"]:
            col_mock = mock.Mock()
            # Now return an int for .sum()
            col_mock.isna.return_value.sum.return_value = 1
            col_mock.impute = mock.MagicMock(return_value=col_mock)
            col_mocks[col] = col_mock

        frame_mock.__getitem__.side_effect = lambda col: col_mocks[col]
        frame_mock.__setitem__.side_effect = lambda col, val: None

        return frame_mock, col_mocks

    train_frame, train_cols = make_mock_frame()
    valid_frame, valid_cols = make_mock_frame()
    test_frame, test_cols = make_mock_frame()

    wrapper.fit(
        train_df=sample_df,
        train_h2o=train_frame,
        valid_h2o=valid_frame,
        test_h2o=test_frame,
    )

    for col in ["age", "income", "gender", "student"]:
        assert train_frame.__setitem__.call_args_list
        assert valid_frame.__setitem__.call_args_list
        assert test_frame.__setitem__.call_args_list


@mock.patch("mlflow.log_artifact")
def test_log_artifacts(mock_log_artifact):
    wrapper = imputation.H2OImputerWrapper()
    wrapper.strategy_map = {"age": "mean", "gender": "mode"}

    wrapper.log(artifact_path="mock_path")

    assert mock_log_artifact.called


@mock.patch("mlflow.artifacts.download_artifacts")
def test_load(mock_download_artifacts):
    mock_path = "tests/mock_strategy_map.json"
    os.makedirs("tests", exist_ok=True)
    with open(mock_path, "w") as f:
        json.dump({"age": "mean", "gender": "mode"}, f)

    mock_download_artifacts.return_value = mock_path

    wrapper = imputation.H2OImputerWrapper.load("run_id")

    assert wrapper.strategy_map == {"age": "mean", "gender": "mode"}
