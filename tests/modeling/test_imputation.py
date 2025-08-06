import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch
from student_success_tool.modeling import imputation  # adjust import
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

    # All original columns should still be in the result
    assert original_cols.issubset(result_cols)

    # All extra columns should be missing flags
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


@patch("mlflow.active_run", return_value=True)
@patch("mlflow.start_run")
@patch("mlflow.log_artifact")
@patch("mlflow.end_run")
def test_pipeline_logged_to_mlflow(
    mock_end_run,
    mock_log_artifact,
    mock_start_run,
    mock_active_run,
    sample_df,
):
    imputer = imputation.SklearnImputerWrapper()
    imputer.fit(sample_df, artifact_path="test_artifact_path")

    mock_end_run.assert_called_once()
    mock_start_run.assert_called_once()
    mock_log_artifact.assert_called_once()
