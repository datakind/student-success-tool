import collections

import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_numeric_dtype
from unittest.mock import patch

from student_success_tool.modeling.inference import (
    _get_mapped_feature_name,
    calculate_shap_values,
    calculate_shap_values_spark_udf,
    select_top_features_for_display,
    generate_ranked_feature_table,
    top_shap_features,
    support_score_distribution_table,
)


class DummyKernelExplainer:
    def __call__(self, X):
        Explanation = collections.namedtuple("Explanation", ["values", "feature_names"])
        return Explanation(np.random.rand(len(X), len(X.columns)) * 0.1, X.columns)

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
        "needs_support_threshold_prob",
        "features_table",
        "exp",
    ],
    [
        (
            pd.DataFrame(
                {
                    "x1": ["val1", "val2", "val3"],
                    "x2": [True, False, True],
                    "x3": [2.0, 1.0001, 0.5],
                    "x4": [1, 2, 3],
                }
            ),
            pd.Series([1, 2, 3]),
            [0.9, 0.1, 0.5],
            np.array(
                [[1.0, 0.9, 0.8, 0.7], [0.0, -1.0, 0.9, -0.8], [0.25, 0.0, -0.5, 0.75]]
            ),
            3,
            0.5,
            {
                "x1": {"name": "feature #1"},
                "x2": {"name": "feature #2"},
                "x3": {"name": "feature #3"},
            },
            pd.DataFrame(
                {
                    "Student ID": [1, 2, 3],
                    "Support Score": [0.9, 0.1, 0.5],
                    "Support Needed": [True, False, True],
                    "Feature_1_Name": ["feature #1", "feature #2", "x4"],
                    "Feature_1_Value": ["val1", "False", "3"],
                    "Feature_1_Importance": [1.0, -1.0, 0.75],
                    "Feature_2_Name": ["feature #2", "feature #3", "feature #3"],
                    "Feature_2_Value": ["True", "1.0", "0.5"],
                    "Feature_2_Importance": [0.9, 0.9, -0.5],
                    "Feature_3_Name": ["feature #3", "x4", "feature #1"],
                    "Feature_3_Value": ["2.0", "2", "val3"],
                    "Feature_3_Importance": [0.8, -0.8, 0.25],
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
            None,
            pd.DataFrame(
                {
                    "Student ID": [1, 2, 3],
                    "Support Score": [0.9, 0.1, 0.5],
                    "Feature_1_Name": ["x1", "x2", "x4"],
                    "Feature_1_Value": ["val1", "False", "3"],
                    "Feature_1_Importance": [1.0, -1.0, 0.75],
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
    needs_support_threshold_prob,
    features_table,
    exp,
):
    obs = select_top_features_for_display(
        features,
        unique_ids,
        predicted_probabilities,
        shap_values,
        n_features=n_features,
        needs_support_threshold_prob=needs_support_threshold_prob,
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


@pytest.fixture
def ranked_feature_table_data():
    features = pd.DataFrame(
        {
            "pell_status": [False, True, False],
            "english_math_gateway": ["E", "M", "M"],
            "term_gpa": [4.0, 3.0, 2.0],
        }
    )
    shap_values = np.array(
        [
            [0.1, -0.2, 0.05],
            [0.3, 0.1, -0.05],
            [0.2, -0.1, 0.15],
        ]
    )
    features_table = {
        "pell_status": {"name": "Pell Status"},
        "english_math_gateway": {"name": "English or Math Gateway"},
        "term_gpa": {"name": "Term GPA"},
    }
    return features, shap_values, features_table


@pytest.mark.parametrize("use_features_table", [True, False])
def test_generate_ranked_feature_table(ranked_feature_table_data, use_features_table):
    features, shap_values, features_table = ranked_feature_table_data

    selected_features_table = features_table if use_features_table else None

    result = generate_ranked_feature_table(
        features, shap_values, selected_features_table
    )

    assert isinstance(result, pd.DataFrame) and not result.empty
    assert set(result.columns) == {
        "Feature Name",
        "Data Type",
        "Average SHAP Magnitude",
    }

    # Verify descending sort order by Average SHAP Magnitude
    assert result["Average SHAP Magnitude"].is_monotonic_decreasing

    if use_features_table:
        assert "English or Math Gateway" in result["Feature Name"].values
    else:
        assert "term_gpa" in result["Feature Name"].values


@pytest.mark.parametrize(
    ["feature_col", "features_table", "exp"],
    [
        (
            "academic_term",
            {"academic_term": {"name": "academic term"}},
            "academic term",
        ),
        ("foo_bar", {"academic_term": {"name": "academic term"}}, "foo_bar"),
        (
            "num_courses_course_subject_area_24",
            {
                r"num_courses_course_subject_area_(\d+)": {
                    "name": "number of courses taken in subject area {} this term"
                }
            },
            "number of courses taken in subject area 24 this term",
        ),
        (
            "num_courses_course_id_engl_101",
            {
                r"num_courses_course_id_(.*)": {
                    "name": "number of times course '{}' taken this term"
                }
            },
            "number of times course 'engl_101' taken this term",
        ),
        (
            "num_courses_course_id_engl_101_cumfrac",
            {
                r"num_courses_course_id_(.*)_cumfrac": {
                    "name": "fraction of times course '{}' taken so far"
                }
            },
            "fraction of times course 'engl_101' taken so far",
        ),
    ],
)
def test_get_mapped_feature_name(feature_col, features_table, exp):
    obs = _get_mapped_feature_name(feature_col, features_table)
    assert isinstance(obs, str)
    assert obs == exp


@pytest.fixture
def sample_data():
    features = pd.DataFrame(
        {
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "feature3": [7, 8, 9],
            "feature4": [0, 1, 0],
            "feature5": [2, 2, 2],
            "feature6": [1, 1, 1],
            "feature7": [0, 0, 1],
            "feature8": [3, 3, 3],
            "feature9": [4, 4, 4],
            "feature10": [5, 5, 5],
            "feature11": [6, 6, 6],
        }
    )
    unique_ids = pd.Series([101, 102, 103])
    shap_values = np.array(
        [
            [0.1, 0.3, 0.2, 0.0, 0.4, 0.1, 0.0, 0.3, 0.2, 0.5, 0.6],
            [0.2, 0.2, 0.1, 0.0, 0.3, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            [0.3, 0.1, 0.3, 0.0, 0.2, 0.0, 0.0, 0.1, 0.4, 0.3, 0.4],
        ]
    )
    features_table = {
        "feature1": {
            "name": "Feature 1 Name",
            "short_desc": "A short description of feature 1",
            "long_desc": "A long description of feature 1",
        },
        "feature2": {
            "name": "Feature 2 Name",
            "short_desc": "A short description of feature 2",
            "long_desc": "A long description of feature 2",
        },
        "feature3": {
            "name": "Feature 3 Name",
        },
    }
    return features, unique_ids, shap_values, features_table


def test_top_shap_features_behavior(sample_data):
    features, unique_ids, shap_values, features_table = sample_data
    result = top_shap_features(
        features, unique_ids, shap_values, features_table=features_table
    )

    # Check output shape and columns
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {
        "student_id",
        "feature_name",
        "shap_value",
        "feature_value",
        "feature_readable_name",
        "feature_short_desc",
        "feature_long_desc",
    }

    # Check top 10 feature selection
    top_features = result["feature_name"].unique()
    assert len(top_features) == 10

    grouped = result.groupby("feature_readable_name")["shap_value"].apply(
        lambda x: np.mean(np.abs(x))
    )
    shap_values = grouped.sort_values(ascending=False).values
    assert all(
        shap_values[i] >= shap_values[i + 1] for i in range(len(shap_values) - 1)
    )

    assert len(grouped) == 10
    print(grouped)
    assert grouped.index[0] == "Feature 1 Name"
    assert grouped.index[1] == "Feature 2 Name"

    assert (
        result["feature_short_desc"]
        .apply(lambda x: isinstance(x, str) or x is None)
        .all()
    )
    assert (
        result["feature_long_desc"]
        .apply(lambda x: isinstance(x, str) or x is None)
        .all()
    )


def test_handles_fewer_than_10_features():
    features = pd.DataFrame(
        {
            "feature1": [1, 2],
            "feature2": [3, 4],
        }
    )
    unique_ids = pd.Series([1, 2])
    shap_values = np.array([[0.5, 0.1], [0.3, 0.4]])

    result = top_shap_features(features, unique_ids, shap_values)
    assert set(result["feature_name"].unique()) == {"feature1", "feature2"}
    assert len(result) == 4  # 2 students × 2 features


def test_empty_input():
    features = pd.DataFrame()
    unique_ids = pd.Series(dtype=int)
    shap_values = np.empty((0, 0))

    with pytest.raises(ValueError):
        top_shap_features(features, unique_ids, shap_values)


@patch("student_success_tool.modeling.inference.select_top_features_for_display")
@pytest.mark.parametrize(
    [
        "features",
        "unique_ids",
        "predicted_probabilities",
        "shap_values",
        "n_features",
        "needs_support_threshold_prob",
        "features_table",
        "exp",
    ],
    [
        (
            pd.DataFrame(
                {
                    "x1": ["val1", "val2", "val3"],
                    "x2": [True, False, True],
                    "x3": [2.0, 1.0001, 0.5],
                    "x4": [1, 2, 3],
                }
            ),
            pd.Series([1, 2, 3]),
            [0.9, 0.1, 0.5],
            np.array(
                [
                    [1.0, 0.9, 0.8, 0.7],
                    [0.0, -1.0, 0.9, -0.8],
                    [0.25, 0.0, -0.5, 0.75],
                ]
            ),
            3,
            0.5,
            {
                "x1": {"name": "feature #1"},
                "x2": {"name": "feature #2"},
                "x3": {"name": "feature #3"},
            },
            pd.DataFrame(
                {
                    "Student ID": [1, 2, 3],
                    "Support Score": [0.9, 0.1, 0.5],
                    "Support Needed": [True, False, True],
                    "Feature_1_Name": ["feature #1", "feature #2", "x4"],
                    "Feature_1_Value": ["val1", "False", "3"],
                    "Feature_1_Importance": [1.0, -1.0, 0.75],
                    "Feature_2_Name": ["feature #2", "feature #3", "feature #3"],
                    "Feature_2_Value": ["True", "1.0", "0.5"],
                    "Feature_2_Importance": [0.9, 0.9, -0.5],
                    "Feature_3_Name": ["feature #3", "x4", "feature #1"],
                    "Feature_3_Value": ["2.0", "2", "val3"],
                    "Feature_3_Importance": [0.8, -0.8, 0.25],
                }
            ),
        )
    ],
)
def test_support_score_distribution_table(
    mock_select_top_features_for_display,
    features,
    unique_ids,
    predicted_probabilities,
    shap_values,
    n_features,
    needs_support_threshold_prob,
    features_table,
    exp,
):
    inference_params = {
        "num_top_features": n_features,
        "min_prob_pos_label": needs_support_threshold_prob or 0.0,
    }

    mock_select_top_features_for_display.return_value = exp

    result = support_score_distribution_table(
        df_serving=features,
        unique_ids=unique_ids,
        pred_probs=predicted_probabilities,
        shap_values=pd.DataFrame(shap_values),
        inference_params=inference_params,
        features_table=features_table,
    )

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {
        "bin_lower",
        "bin_upper",
        "support_score",
        "count_of_students",
        "pct",
    }
    assert result["count_of_students"].sum() == len(unique_ids)
    assert np.isclose(result["pct"].sum(), 100.0, atol=0.01)

    # Binning logic checks
    for _, row in result.iterrows():
        expected_midpoint = round((row["bin_lower"] + row["bin_upper"]) / 2, 2)
        assert row["support_score"] == expected_midpoint
        assert round(row["bin_upper"] - row["bin_lower"], 2) == 0.1
        assert 0.1 <= row["bin_lower"] < row["bin_upper"] <= 1.0
