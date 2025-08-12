import pandas as pd
import numpy as np
import os
import pytest
from unittest import mock
from student_success_tool.modeling.h2o_modeling import imputation
from sklearn.pipeline import Pipeline


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "num_low_skew": [1.0, 2.0, np.nan, 4.0],
            "num_high_skew": [1, 1000, np.nan, 3],
            "bool_col": [True, False, None, True],
            "cat_col": ["a", "b", None, "a"],
            "text_col": ["x", None, "y", "x"],
            "complete_col": [10, 20, 30, 40],
        },
        index=["s1", "s2", "s3", "s4"],
    )


def test_fit_and_transform_shapes_and_columns(sample_df):
    imputer = imputation.SklearnImputerWrapper()
    imputer.fit(sample_df)
    result = imputer.transform(sample_df)

    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == sample_df.shape[0]

    original_cols = set(sample_df.columns)
    result_cols = set(result.columns)

    assert original_cols.issubset(result_cols)

    extra_cols = result_cols - original_cols
    assert all(col.endswith("_missing_flag") for col in extra_cols)


def test_transform_raises_if_not_fitted(sample_df):
    imputer = imputation.SklearnImputerWrapper()
    with pytest.raises(ValueError, match="Pipeline not fitted"):
        imputer.transform(sample_df)


def test_missing_values_filled(sample_df):
    imputer = imputation.SklearnImputerWrapper()
    imputer.fit(sample_df)
    result = imputer.transform(sample_df)

    assert result.isnull().sum().sum() == 0  # No missing values remain


def test_pipeline_instance_and_step_names(sample_df):
    imputer = imputation.SklearnImputerWrapper()
    pipeline = imputer.fit(sample_df)
    assert isinstance(pipeline, Pipeline)
    assert "imputer" in dict(pipeline.named_steps)


def test_missing_flags_added_only_for_missing_columns(sample_df):
    imputer = imputation.SklearnImputerWrapper()
    imputer.fit(sample_df)
    result = imputer.transform(sample_df)

    expected_flags = {
        "num_low_skew_missing_flag",
        "num_high_skew_missing_flag",
        "bool_col_missing_flag",
        "cat_col_missing_flag",
        "text_col_missing_flag",
    }

    for flag_col in expected_flags:
        assert flag_col in result.columns
        assert set(result[flag_col].unique()).issubset({0, 1})

    assert "complete_col_missing_flag" not in result.columns


@mock.patch("mlflow.active_run")
@mock.patch("mlflow.start_run")
@mock.patch("mlflow.log_artifact")
@mock.patch("mlflow.end_run")
def test_pipeline_logged_to_mlflow(
    mock_end_run,
    mock_log_artifact,
    mock_start_run,
    mock_active_run,
    sample_df,
):
    imputer = imputation.SklearnImputerWrapper()
    imputer.fit(sample_df)

    imputer.log_pipeline(artifact_path="test_artifact_path")

    # Expect pipeline, input_dtypes, input_feature_names, missing_flag_cols to be logged
    assert mock_log_artifact.call_count == 4

    artifact_paths = [
        call.kwargs["artifact_path"] for call in mock_log_artifact.call_args_list
    ]
    assert all(path == "test_artifact_path" for path in artifact_paths)

    logged_filenames = [
        os.path.basename(call.args[0]) for call in mock_log_artifact.call_args_list
    ]
    assert "imputer_pipeline.joblib" in logged_filenames
    assert "input_dtypes.json" in logged_filenames
    assert "input_feature_names.json" in logged_filenames
