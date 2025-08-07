import pandas as pd
import numpy as np
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


def test_missing_flags_added_only_for_missing_columns(sample_df):
    imputer = imputation.SklearnImputerWrapper()
    imputer.fit(sample_df)
    result = imputer.transform(sample_df)

    # These columns have missing data
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

    # This column has no missingness, should not have flag
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

    # Now call the method that triggers MLflow logging
    imputer.log_pipeline(artifact_path="test_artifact_path")

    # Assert MLflow behavior
    mock_log_artifact.assert_called_once()
