import pandas as pd
import numpy as np
import pytest
from unittest import mock
from student_success_tool.modeling.h2o_modeling import inference


# Existing SHAP grouping test
def test_group_shap_by_feature_basic():
    df = pd.DataFrame(
        {
            "gender.M": [0.1, 0.2],
            "gender.F": [0.3, 0.4],
            "age": [0.5, -0.1],
            "BiasTerm": [0.0, 0.0],
        }
    )
    grouped = inference.group_shap_values(df, drop_bias_term=True)
    expected = pd.DataFrame({"gender": [0.4, 0.6], "age": [0.5, -0.1]})
    pd.testing.assert_frame_equal(grouped, expected)


# One-hot with missing flag grouping
def test_group_feature_values_with_missing_flag():
    df = pd.DataFrame(
        {
            "ethnicity.Asian": ["Asian", None],
            "ethnicity.Black": [None, "Black"],
            "ethnicity_missing_flag": [False, True],
            "income": [50000, 60000],
        }
    )
    grouped = inference.group_feature_values(df, group_missing_flags=True)
    expected = pd.DataFrame(
        {
            "ethnicity": ["Asian", "MISSING"],
            "income": [50000, 60000],
        }
    )
    pd.testing.assert_frame_equal(grouped, expected)


# New test: ambiguous encoding should raise ValueError
def test_group_feature_values_ambiguous_encoding_raises():
    df = pd.DataFrame(
        {
            "ethnicity.Asian": ["Asian", "Asian"],
            "ethnicity.Black": ["Black", None],
        }
    )
    with pytest.raises(ValueError, match="Could not resolve base feature"):
        inference.group_feature_values(df, group_missing_flags=True)


def test_create_color_hint_features_mixed_types():
    original_df = pd.DataFrame(
        {
            "gender": pd.Series(["M", "F"], dtype="category"),
            "income": [50000, 60000],
            "opted_in": [True, False],
        }
    )
    orig_dtypes = {"gender": "category", "income": "int", "opted_in": "bool"}
    grouped_df = pd.DataFrame(
        {"gender": [1.0, 0.0], "income": [0.3, 0.7], "opted_in": [1, 0]}
    )
    result = inference.create_color_hint_features(
        grouped_df, original_dtypes=orig_dtypes
    )
    assert result["gender"].tolist() == ["1.0", "0.0"]
    assert result["income"].tolist() == [0.3, 0.7]
    assert result["opted_in"].tolist() == [1, 0]


@mock.patch("student_success_tool.modeling.h2o_modeling.inference.mlflow.log_figure")
@mock.patch("student_success_tool.modeling.h2o_modeling.inference.shap.summary_plot")
def test_plot_grouped_shap_calls_summary_plot(mock_summary_plot, mock_log_figure):
    contribs_df = pd.DataFrame(
        {"feature.X.1": [0.1, 0.2], "feature.X.2": [0.2, 0.3], "feature.Y": [0.3, 0.1]}
    )
    original_df = pd.DataFrame({"feature.X": ["A", "B"], "feature.Y": ["C", "D"]})
    inference.plot_grouped_shap(contribs_df, original_df, group_missing_flags=False)
    mock_summary_plot.assert_called_once()


@mock.patch("student_success_tool.modeling.h2o_modeling.inference.h2o.H2OFrame")
def test_compute_h2o_shap_contributions_with_bias_drop(mock_h2o_frame):
    mock_model = mock.MagicMock()
    mock_model.predict_contributions.return_value.as_data_frame.return_value = (
        pd.DataFrame(
            {"feature1": [0.1, 0.2], "feature2": [0.3, 0.4], "BiasTerm": [0.5, 0.6]}
        )
    )
    mock_model._model_json = {"output": {"names": ["feature1", "feature2", "target"]}}

    h2o_frame = mock.MagicMock()
    h2o_frame.nrows = 2
    h2o_frame.__getitem__.return_value.nrows = 2
    h2o_frame.__getitem__.return_value.as_data_frame.return_value = pd.DataFrame(
        {"feature1": [1, 2], "feature2": [3, 4]}
    )

    contribs, inputs = inference.compute_h2o_shap_contributions(
        mock_model,
        h2o_frame,
        drop_bias=True,
        return_features=True,
    )

    assert "BiasTerm" not in contribs.columns
    assert list(inputs.columns) == ["feature1", "feature2"]

    contribs = inference.compute_h2o_shap_contributions(
        mock_model,
        h2o_frame,
        drop_bias=True,
    )
    assert "BiasTerm" not in contribs.columns
    assert contribs.shape == (2, 2)


def test_group_missing_flags_aggregated_correctly():
    df = pd.DataFrame(
        {
            "math_placement.M": [0.1, 0.2],
            "math_placement.F": [0.3, 0.1],
            "math_placement_missing_flag": [0.4, 0.7],
            "income": [0.5, -0.1],
            "BiasTerm": [0.0, 0.0],
        }
    )

    grouped_with_flag = inference.group_shap_values(
        df, drop_bias_term=True, group_missing_flags=True
    )
    expected_with_flag = pd.DataFrame(
        {"math_placement": [0.8, 1.0], "income": [0.5, -0.1]}
    )
    pd.testing.assert_frame_equal(grouped_with_flag, expected_with_flag)

    grouped_without_flag = inference.group_shap_values(
        df, drop_bias_term=True, group_missing_flags=False
    )
    expected_without_flag = pd.DataFrame(
        {
            "math_placement": [0.4, 0.3],
            "math_placement_missing_flag": [0.4, 0.7],
            "income": [0.5, -0.1],
        }
    )
    pd.testing.assert_frame_equal(grouped_without_flag, expected_without_flag)


def test_create_color_hint_features_keeps_missing_flags_numeric():
    # grouped_df after grouping (e.g., SHAP + features grouped to base names)
    grouped_df = pd.DataFrame(
        {
            "gender": [1.0, 0.0],  # categorical in original data
            "income": [0.3, 0.7],  # numeric in original data
            "gpa_missing_flag": [1, 0],  # boolean flag (not present in original_dtypes)
        }
    )

    # original dtypes from raw (pre-imputation) – intentionally no *_missing_flag here
    orig_dtypes = {
        "gender": "category",
        "income": "float64",
        # "gpa_missing_flag" intentionally omitted
    }

    result = inference.create_color_hint_features(
        grouped_df, original_dtypes=orig_dtypes
    )

    # gender => categorical → stringified values
    assert list(result["gender"]) == ["1.0", "0.0"]
    # income => numeric → unchanged float values
    assert list(result["income"]) == [0.3, 0.7]
    # *_missing_flag not in original_dtypes => stays numeric/boolean (NOT stringified)
    assert list(result["gpa_missing_flag"]) == [1, 0]


@mock.patch("student_success_tool.modeling.h2o_modeling.inference.mlflow.log_figure")
@mock.patch("student_success_tool.modeling.h2o_modeling.inference.shap.summary_plot")
def test_plot_grouped_shap_works_without_original_dtypes(
    mock_summary_plot, mock_log_figure
):
    # Minimal contribs and features where grouped names align
    contribs_df = pd.DataFrame(
        {
            "feature.X.A": [0.1, 0.2],
            "feature.X.B": [0.2, 0.3],
            "feature.Y": [0.3, 0.1],
        }
    )
    features_df = pd.DataFrame(
        {
            "feature.X.A": [1, 0],
            "feature.X.B": [0, 1],
            "feature.Y": [1.2, 2.3],
        }
    )

    # Call without original_dtypes — should fall back to grouped_feats (no color hint)
    inference.plot_grouped_shap(
        contribs_df,
        features_df,
        group_missing_flags=True,  # exercise grouping path too
        original_dtypes=None,
        max_display=10,
        mlflow_name="dummy.png",
    )

    # Ensure SHAP and MLflow were called
    mock_summary_plot.assert_called_once()
    mock_log_figure.assert_called_once()


# -------------------------------
# Tests for inference.predict_h2o
# -------------------------------

@mock.patch("student_success_tool.modeling.h2o_modeling.utils._to_h2o", return_value="hf")
def test_predict_h2o_with_dataframe(mock_to_h2o):
    df = pd.DataFrame({"a": [1, 2], "b_missing_flag": [0, 1]})
    result = inference.predict_h2o(df)
    mock_to_h2o.assert_called_once_with(df, force_enum_cols=["b_missing_flag"])
    assert result == "hf"


@mock.patch("student_success_tool.modeling.h2o_modeling.utils._to_h2o", return_value="hf")
def test_predict_h2o_with_ndarray_and_feature_names(mock_to_h2o):
    arr = np.array([[1, 2], [3, 4]])
    feature_names = ["x", "y_missing_flag"]
    result = inference.predict_h2o(arr, feature_names=feature_names)

    mock_to_h2o.assert_called_once()
    called_df, kwargs = mock_to_h2o.call_args
    assert list(called_df[0].columns) == feature_names
    assert kwargs["force_enum_cols"] == ["y_missing_flag"]
    assert result == "hf"


def test_predict_h2o_with_ndarray_missing_feature_names_raises():
    arr = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="feature_names must be provided"):
        inference.predict_h2o(arr)


# -------------------------------
# Tests for predict_probs_h2o
# -------------------------------

@mock.patch("student_success_tool.modeling.h2o_modeling.utils._to_pandas")
@mock.patch("student_success_tool.modeling.h2o_modeling.inference.predict_h2o", return_value="hf")
def test_predict_probs_h2o_returns_all_probs_when_no_pos_label(mock_predict_h2o, mock_to_pandas):
    df = pd.DataFrame({"f1": [1], "f2_missing_flag": [0]})

    mock_model = mock.MagicMock()
    mock_model.predict.return_value = "raw_pred"

    mock_to_pandas.return_value = pd.DataFrame(
        {"predict": ["a"], "A": [0.2], "B": [0.8]}
    )

    result = inference.predict_probs_h2o(df, mock_model)

    np.testing.assert_array_equal(result, np.array([[0.2, 0.8]]))
    mock_model.predict.assert_called_once_with("hf")


@mock.patch("student_success_tool.modeling.h2o_modeling.utils._to_pandas")
@mock.patch("student_success_tool.modeling.h2o_modeling.inference.predict_h2o", return_value="hf")
def test_predict_probs_h2o_with_pos_label(mock_predict_h2o, mock_to_pandas):
    df = pd.DataFrame({"f1": [1], "f2": [2]})
    mock_model = mock.MagicMock()

    mock_to_pandas.return_value = pd.DataFrame(
        {"predict": ["a"], "0": [0.1], "1": [0.9]}
    )

    result = inference.predict_probs_h2o(df, mock_model, pos_label=1)

    np.testing.assert_array_equal(result, np.array([0.9]))


@mock.patch("student_success_tool.modeling.h2o_modeling.utils._to_pandas")
@mock.patch("student_success_tool.modeling.h2o_modeling.inference.predict_h2o", return_value="hf")
def test_predict_probs_h2o_with_missing_pos_label_raises(mock_predict_h2o, mock_to_pandas):
    df = pd.DataFrame({"f1": [1]})
    mock_model = mock.MagicMock()

    mock_to_pandas.return_value = pd.DataFrame(
        {"predict": ["a"], "A": [0.2], "B": [0.8]}
    )

    with pytest.raises(ValueError, match="pos_label X not found"):
        inference.predict_probs_h2o(df, mock_model, pos_label="X")