import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_numeric_dtype

from student_success_tool.modeling.inference import (
    calculate_shap_values,
    calculate_shap_values_spark_udf,
    select_top_features_for_display,
)


class DummyKernelExplainer:
    def shap_values(self, X):
        # for simplicity, return random numbers of the proper shape
        return np.random.rand(len(X), len(X.columns)) * 0.1


@pytest.fixture(scope="module")
def explainer():
    return DummyKernelExplainer()


@pytest.mark.parametrize(
    [
        "features",
        "unique_ids",
        "predicted_probabilities",
        "shap_values",
        "n_features",
        "features_table",
        "exp",
    ],
    [
        (
            pd.DataFrame(
                {
                    "x1": ["val1", "val2", "val3"],
                    "x2": [True, False, True],
                    "x3": [2.0, 1.0, 0.5],
                    "x4": [1, 2, 3],
                }
            ),
            pd.Series([1, 2, 3]),
            [0.9, 0.1, 0.5],
            np.array(
                [[1.0, 0.9, 0.8, 0.7], [0.0, -1.0, 0.9, -0.8], [0.25, 0.0, -0.5, 0.75]]
            ),
            3,
            {
                "x1": {"name": "feature #1"},
                "x2": {"name": "feature #2"},
                "x3": {"name": "feature #3"},
            },
            pd.DataFrame(
                {
                    "Student ID": [1, 1, 1, 2, 2, 2, 3, 3, 3],
                    "Support Score": [0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5],
                    "Top Indicators": [
                        "feature #1",
                        "feature #2",
                        "feature #3",
                        "feature #2",
                        "feature #3",
                        "x4",
                        "x4",
                        "feature #3",
                        "feature #1",
                    ],
                    "Indicator Value": [
                        "val1",
                        True,
                        2.0,
                        False,
                        1.0,
                        2,
                        3,
                        0.5,
                        "val3",
                    ],
                    "SHAP Value": [1.0, 0.9, 0.8, -1.0, 0.9, -0.8, 0.75, -0.5, 0.25],
                    "Rank": [1, 2, 3, 1, 2, 3, 1, 2, 3],
                }
            ),
        ),
        (
            pd.DataFrame(
                {
                    "x1": ["val1", "val2", "val3"],
                    "x2": [True, False, True],
                    "x3": [2.0, 1.0, 0.5],
                    "x4": [1, 2, 3],
                }
            ),
            pd.Series([1, 2, 3]),
            [0.9, 0.1, 0.5],
            np.array(
                [[1.0, 0.9, 0.8, 0.7], [0.0, -1.0, 0.9, -0.8], [0.25, 0.0, -0.5, 0.75]]
            ),
            1,
            None,
            pd.DataFrame(
                {
                    "Student ID": [1, 2, 3],
                    "Support Score": [0.9, 0.1, 0.5],
                    "Top Indicators": ["x1", "x2", "x4"],
                    "Indicator Value": ["val1", False, 3],
                    "SHAP Value": [1.0, -1.0, 0.75],
                    "Rank": [1, 1, 1],
                }
            ),
        ),
    ],
)
def test_select_top_features_for_display(
    features,
    unique_ids,
    predicted_probabilities,
    shap_values,
    n_features,
    features_table,
    exp,
):
    obs = select_top_features_for_display(
        features,
        unique_ids,
        predicted_probabilities,
        shap_values,
        n_features=n_features,
        features_table=features_table,
    )
    assert isinstance(obs, pd.DataFrame) and not obs.empty
    assert pd.testing.assert_frame_equal(obs, exp) is None


@pytest.mark.parametrize(
    ["df", "feature_names", "fillna_values", "student_id_col", "exp_shape"],
    [
        (
            pd.DataFrame(
                {
                    "student_id": [1, 2, 3],
                    "feature1": [0.1, 0.2, 0.3],
                    "feature2": [0.4, 0.5, 0.6],
                }
            ),
            ["feature1", "feature2"],
            pd.Series([0.2, 0.5]),
            "student_id",
            (3, 3),
        ),
    ],
)
def test_calculate_shap_values(
    explainer, df, feature_names, fillna_values, student_id_col, exp_shape
):
    obs = calculate_shap_values(
        df,
        explainer,
        feature_names=feature_names,
        fillna_values=fillna_values,
        student_id_col=student_id_col,
    )
    assert isinstance(obs, pd.DataFrame) and not obs.empty
    assert obs.shape == exp_shape
    assert student_id_col in obs.columns
    assert all(is_numeric_dtype(obs[feature_name]) for feature_name in feature_names)
    assert obs[student_id_col].equals(df[student_id_col])


@pytest.mark.parametrize(
    "input_data, expected_shape",
    [
        (
            {
                "student_id": [1, 2, 3],
                "feature1": [0.1, 0.2, 0.3],
                "feature2": [0.4, 0.5, 0.6],
            },
            (3, 3),
        ),
        (
            {"student_id": [1, 2], "feature1": [0.1, 0.2], "feature2": [0.4, 0.5]},
            (2, 3),
        ),
    ],
)
def test_calculate_shap_values_spark_udf_basic(input_data, expected_shape, explainer):
    df = pd.DataFrame(input_data)
    student_id_col = "student_id"
    model_features = ["feature1", "feature2"]
    mode = df.mode().iloc[0]

    iterator = iter([df])

    result = list(
        calculate_shap_values_spark_udf(
            iterator,
            student_id_col=student_id_col,
            model_features=model_features,
            explainer=explainer,
            mode=mode,
        )
    )

    # Check that the result contains the expected number of rows and columns
    shap_df = result[0]
    assert shap_df.shape == expected_shape

    # Ensure that 'student_id' column is present
    assert student_id_col in shap_df.columns

    # Ensure that SHAP values are generated and are numeric
    assert is_numeric_dtype(shap_df[model_features].iloc[0, 0])
    assert is_numeric_dtype(shap_df[model_features].iloc[0, 1])

    # Ensure student IDs are correctly reattached
    assert shap_df[student_id_col].iloc[0] == 1
    assert shap_df[student_id_col].iloc[1] == 2


@pytest.mark.parametrize(
    "batch1_data, batch2_data, expected_shape1, expected_shape2",
    [
        (
            {
                "student_id": [1, 2, 3],
                "feature1": [0.1, 0.2, 0.3],
                "feature2": [0.4, 0.5, 0.6],
            },
            {
                "student_id": [4, 5, 6],
                "feature1": [0.7, 0.8, 0.9],
                "feature2": [0.6, 0.7, 0.8],
            },
            (3, 3),
            (3, 3),
        ),
        (
            {
                "student_id": [4, 5, 6],
                "feature1": [0.1, 0.2, 0.3],
                "feature2": [0.4, 0.5, 0.6],
            },
            {
                "student_id": [4, 5, 6],
                "feature1": [0.5, 0.6, 0.7],
                "feature2": [0.7, 0.8, 0.9],
            },
            (3, 3),
            (3, 3),
        ),
    ],
)
def test_calculate_shap_values_spark_udf_multiple_batches(
    batch1_data, batch2_data, expected_shape1, expected_shape2, explainer
):
    batch1 = pd.DataFrame(batch1_data)
    batch2 = pd.DataFrame(batch2_data)

    student_id_col = "student_id"
    model_features = ["feature1", "feature2"]
    mode = batch1.mode().iloc[0]

    iterator = iter([batch1, batch2])

    result = list(
        calculate_shap_values_spark_udf(
            iterator,
            student_id_col=student_id_col,
            model_features=model_features,
            explainer=explainer,
            mode=mode,
        )
    )

    # Ensure we have two DataFrames
    assert len(result) == 2

    # Check first batch
    shap_df1 = result[0]
    assert shap_df1.shape == expected_shape1

    # Check second batch
    shap_df2 = result[1]
    assert shap_df2.shape == expected_shape2
