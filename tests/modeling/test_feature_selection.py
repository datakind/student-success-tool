import numpy as np
import pandas as pd
import pytest

from student_success_tool.modeling.feature_selection import (
    drop_collinear_features_iteratively,
    drop_incomplete_features,
    drop_low_variance_features,
)


@pytest.mark.parametrize(
    "threshold,expected_columns",
    [
        (0.4, ["complete_feature"]),
        (0.5, ["complete_feature", "kinda_incomplete_feature"]),
    ],
)
def test_drop_incomplete_features(threshold, expected_columns):
    test_features = pd.DataFrame(
        {
            "complete_feature": [1, 2, 3, 4, 5],
            "kinda_incomplete_feature": [np.nan, np.nan, 3, 4, 5],
            "incomplete_feature": [np.nan, 2, np.nan, np.nan, np.nan],
        }
    )
    returned_df = drop_incomplete_features(test_features, threshold=threshold)
    assert set(returned_df.columns.values) == set(expected_columns)
    assert returned_df.shape[0] == test_features.shape[0]


def test_drop_low_variance_features():
    low_variance_features = pd.DataFrame(
        {"has_variance": [1, 2, 3, 4, 5], "no_variance": [1, 1, 1, 1, 1]}
    )
    returned_df = drop_low_variance_features(low_variance_features, threshold=0.0)
    assert set(returned_df.columns.values) == {"has_variance"}
    assert returned_df.shape[0] == low_variance_features.shape[0]


def test_drop_low_variance_features_no_dropped_columns_incl_categorical():
    low_variance_features = pd.DataFrame(
        {
            "has_variance": [1, 2, 3, 4, 5],
            "categorical": ["cat", "eg", "or", "ic", "al"],
        }
    )
    returned_df = drop_low_variance_features(low_variance_features, threshold=0.0)
    assert set(returned_df.columns.values) == set(low_variance_features.columns.values)
    assert returned_df.shape[0] == low_variance_features.shape[0]


def test_drop_collinear_features_iteratively():
    related_features = pd.DataFrame(
        {"A": [1, 2, 3, 4, 5], "B": [6, 7, 8, 9, 10], "C": [6.5, 1.5, 4, 3.5, 2]}
    )
    returned_df = drop_collinear_features_iteratively(
        related_features, force_include_cols=[]
    )
    assert set(returned_df.columns.values) == {"A", "C"}
    assert returned_df.shape[0] == related_features.shape[0]


def test_drop_collinear_features_iteratively_force_include():
    related_features = pd.DataFrame(
        {"A": [1, 2, 3, 4, 5], "B": [6, 7, 8, 9, 10], "C": [6.5, 1.5, 4, 3.5, 2]}
    )
    returned_df = drop_collinear_features_iteratively(
        related_features, force_include_cols=["B"]
    )
    assert set(returned_df.columns.values) == {"B", "C"}
    assert returned_df.shape[0] == related_features.shape[0]


def test_drop_collinear_features_iteratively_fails_with_one_feature_left():
    related_features = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [6, 7, 8, 9, 10]})
    with pytest.raises(Exception):
        drop_collinear_features_iteratively(related_features)
