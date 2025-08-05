import pandas as pd
import unittest.mock as mock

from student_success_tool.modeling import inference_h2o


def test_group_shap_by_feature_basic():
    df = pd.DataFrame(
        {
            "gender.M": [0.1, 0.2],
            "gender.F": [0.3, 0.4],
            "age": [0.5, -0.1],
            "BiasTerm": [0.0, 0.0],
        }
    )
    grouped = inference_h2o.group_shap_by_feature(df)
    expected = pd.DataFrame({"gender": [0.4, 0.6], "age": [0.5, -0.1]})
    pd.testing.assert_frame_equal(grouped, expected)


def test_group_feature_values_by_feature():
    df = pd.DataFrame(
        {"ethnicity.Asian": [1, 0], "ethnicity.Black": [0, 1], "income": [50000, 60000]}
    )
    grouped = inference_h2o.group_feature_values_by_feature(df)
    expected = pd.DataFrame({"ethnicity": [1, 1], "income": [50000, 60000]})
    pd.testing.assert_frame_equal(grouped, expected)


def test_create_color_hint_features_mixed_types():
    orig_df = pd.DataFrame(
        {
            "gender": pd.Series(["M", "F"], dtype="category"),
            "income": [50000, 60000],
            "opted_in": [True, False],
        }
    )
    grouped_df = pd.DataFrame(
        {"gender": [1.0, 0.0], "income": [0.3, 0.7], "opted_in": [1, 0]}
    )
    result = inference_h2o.create_color_hint_features(orig_df, grouped_df)

    assert result["gender"].tolist() == ["category", "category"]
    assert result["income"].tolist() == [0.3, 0.7]
    assert result["opted_in"].tolist() == [1, 0]


@mock.patch("student_success_tool.modeling.inference_h2o.shap.summary_plot")
def test_plot_grouped_shap_calls_summary_plot(mock_summary_plot):
    contribs_df = pd.DataFrame(
        {"feature.X.1": [0.1, 0.2], "feature.X.2": [0.2, 0.3], "feature.Y": [0.3, 0.1]}
    )
    input_df = pd.DataFrame(
        {"feature.X.1": [1, 0], "feature.X.2": [0, 1], "feature.Y": [1, 0]}
    )
    original_df = pd.DataFrame({"feature.X": ["A", "B"], "feature.Y": ["C", "D"]})

    inference_h2o.plot_grouped_shap(contribs_df, input_df, original_df)
    assert mock_summary_plot.called


@mock.patch("student_success_tool.modeling.inference_h2o.h2o.H2OFrame")
def test_compute_h2o_shap_contributions_with_bias_drop(mock_h2o_frame):
    mock_model = mock.MagicMock()
    mock_model.predict_contributions.return_value.as_data_frame.return_value = (
        pd.DataFrame(
            {"feature1": [0.1, 0.2], "feature2": [0.3, 0.4], "BiasTerm": [0.5, 0.6]}
        )
    )
    mock_model._model_json = {"output": {"names": ["feature1", "feature2", "target"]}}

    h2o_frame = mock.MagicMock()
    h2o_frame.__getitem__.return_value.as_data_frame.return_value = pd.DataFrame(
        {"feature1": [1, 2], "feature2": [3, 4]}
    )

    contribs, inputs = inference_h2o.compute_h2o_shap_contributions(
        mock_model, h2o_frame, drop_bias=True
    )

    assert "BiasTerm" not in contribs.columns
    assert list(inputs.columns) == ["feature1", "feature2"]
